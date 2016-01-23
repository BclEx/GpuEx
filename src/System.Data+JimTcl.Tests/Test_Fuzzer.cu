// Code for a demonstration virtual table that generates variations on an input word at increasing edit distances from the original.
//
// A fuzzer virtual table is created like this:
//
//     CREATE VIRTUAL TABLE f USING fuzzer(<fuzzer-data-table>);
//
// When it is created, the new fuzzer table must be supplied with the name of a "fuzzer data table", which must reside in the same database
// file as the new fuzzer table. The fuzzer data table contains the various transformations and their costs that the fuzzer logic uses to generate variations.
//
// The fuzzer data table must contain exactly four columns (more precisely, the statement "SELECT * FROM <fuzzer_data_table>" must return records
// that consist of four columns). It does not matter what the columns are named. 
//
// Each row in the fuzzer data table represents a single character transformation. The left most column of the row (column 0) contains an
// integer value - the identifier of the ruleset to which the transformation rule belongs (see "MULTIPLE RULE SETS" below). The second column of the
// row (column 0) contains the input character or characters. The third column contains the output character or characters. And the fourth column
// contains the integer cost of making the transformation. For example:
//
//    CREATE TABLE f_data(ruleset, cFrom, cTo, Cost);
//    INSERT INTO f_data(ruleset, cFrom, cTo, Cost) VALUES(0, '', 'a', 100);
//    INSERT INTO f_data(ruleset, cFrom, cTo, Cost) VALUES(0, 'b', '', 87);
//    INSERT INTO f_data(ruleset, cFrom, cTo, Cost) VALUES(0, 'o', 'oe', 38);
//    INSERT INTO f_data(ruleset, cFrom, cTo, Cost) VALUES(0, 'oe', 'o', 40);
//
// The first row inserted into the fuzzer data table by the SQL script above indicates that the cost of inserting a letter 'a' is 100.  (All 
// costs are integers.  We recommend that costs be scaled so that the average cost is around 100.) The second INSERT statement creates a rule
// saying that the cost of deleting a single letter 'b' is 87.  The third and fourth INSERT statements mean that the cost of transforming a
// single letter "o" into the two-letter sequence "oe" is 38 and that the cost of transforming "oe" back into "o" is 40.
//
// The contents of the fuzzer data table are loaded into main memory when a fuzzer table is first created, and may be internally reloaded by the
// system at any subsequent time. Therefore, the fuzzer data table should be populated before the fuzzer table is created and not modified thereafter.
// If you do need to modify the contents of the fuzzer data table, it is recommended that the associated fuzzer table be dropped, the fuzzer data
// table edited, and the fuzzer table recreated within a single transaction. Alternatively, the fuzzer data table can be edited then the database
// connection can be closed and reopened.
//
// Once it has been created, the fuzzer table can be queried as follows:
//
//    SELECT word, distance FROM f
//     WHERE word MATCH 'abcdefg'
//       AND distance<200;
//
// This first query outputs the string "abcdefg" and all strings that can be derived from that string by appling the specified transformations.
// The strings are output together with their total transformation cost (called "distance") and appear in order of increasing cost.  No string
// is output more than once.  If there are multiple ways to transform the target string into the output string then the lowest cost transform is
// the one that is returned.  In the example, the search is limited to strings with a total distance of less than 200.
//
// The fuzzer is a read-only table.  Any attempt to DELETE, INSERT, or UPDATE on a fuzzer table will throw an error.
//
// It is important to put some kind of a limit on the fuzzer output.  This can be either in the form of a LIMIT clause at the end of the query,
// or better, a "distance<NNN" constraint where NNN is some number.  The running time and memory requirement is exponential in the value of NNN 
// so you want to make sure that NNN is not too big.  A value of NNN that is about twice the average transformation cost seems to give good results.
//
// The fuzzer table can be useful for tasks such as spelling correction. Suppose there is a second table vocabulary(w) where the w column contains
// all correctly spelled words.   Let $word be a word you want to look up.
//
//   SELECT vocabulary.w FROM f, vocabulary
//    WHERE f.word MATCH $word
//      AND f.distance<=200
//      AND f.word=vocabulary.w
//    LIMIT 20
//
// The query above gives the 20 closest words to the $word being tested. (Note that for good performance, the vocubulary.w column should be indexed.)
//
// A similar query can be used to find all words in the dictionary that begin with some prefix $prefix:
//
//   SELECT vocabulary.w FROM f, vocabulary
//    WHERE f.word MATCH $prefix
//      AND f.distance<=200
//      AND vocabulary.w BETWEEN f.word AND (f.word || x'F7BFBFBF')
//    LIMIT 50
//
// This last query will show up to 50 words out of the vocabulary that match or nearly match the $prefix.
//
// MULTIPLE RULE SETS
//
// Normally, the "ruleset" value associated with all character transformations in the fuzzer data table is zero. However, if required, the fuzzer table
// allows multiple rulesets to be defined. Each query uses only a single ruleset. This allows, for example, a single fuzzer table to support multiple languages.
//
// By default, only the rules from ruleset 0 are used. To specify an alternative ruleset, a "ruleset = ?" expression must be added to the
// WHERE clause of a SELECT, where ? is the identifier of the desired ruleset. For example:
//
//   SELECT vocabulary.w FROM f, vocabulary
//    WHERE f.word MATCH $word
//      AND f.distance<=200
//      AND f.word=vocabulary.w
//      AND f.ruleset=1  -- Specify the ruleset to use here
//    LIMIT 20
//
// If no "ruleset = ?" constraint is specified in the WHERE clause, ruleset 0 is used.
//
// LIMITS
//
// The maximum ruleset number is 2147483647.  The maximum length of either of the strings in the second or third column of the fuzzer data table
// is 50 bytes.  The maximum cost on a rule is 1000.

// If SQLITE_DEBUG is not defined, disable assert statements.
#if !defined(NDEBUG) && !defined(_DEBUG)
#define NDEBUG
#endif
#include <Core+Vdbe\Core+Vdbe.cu.h>

#ifndef OMIT_VIRTUALTABLE
// Forward declaration of objects used by this implementation
typedef struct fuzzer_vtab fuzzer_vtab;
typedef struct fuzzer_cursor fuzzer_cursor;
typedef struct fuzzer_rule fuzzer_rule;
typedef struct fuzzer_seen fuzzer_seen;
typedef struct fuzzer_stem fuzzer_stem;

// Various types.
// fuzzer_cost is the "cost" of an edit operation.
// fuzzer_len is the length of a matching string.  
// fuzzer_ruleid is an ruleset identifier.
typedef int fuzzer_cost;
typedef signed char fuzzer_len;
typedef int fuzzer_ruleid;

// Limits
#define FUZZER_MX_LENGTH           50   // Maximum length of a rule string
#define FUZZER_MX_RULEID   2147483647   // Maximum rule ID
#define FUZZER_MX_COST           1000   // Maximum single-rule cost
#define FUZZER_MX_OUTPUT_LENGTH   100   // Maximum length of an output string

// Each transformation rule is stored as an instance of this object. All rules are kept on a linked list sorted by rCost.
struct fuzzer_rule
{
	fuzzer_rule *Next;			// Next rule in order of increasing rCost
	char *From;					// Transform from
	fuzzer_cost Cost;			// Cost of this transformation
	fuzzer_len FromLength, ToLength; // Length of the zFrom and zTo strings
	fuzzer_ruleid Ruleset;		// The rule set to which this rule belongs
	char To[4];					// Transform to (extra space appended)
};

// A stem object is used to generate variants.  It is also used to record previously generated outputs.
// Every stem is added to a hash table as it is output.  Generation of duplicate stems is suppressed.
// Active stems (those that might generate new outputs) are kepts on a linked list sorted by increasing cost.  The cost is the sum of rBaseCost and pRule->rCost.
struct fuzzer_stem
{
	char *Basis;				// Word being fuzzed
	const fuzzer_rule *Rule;	// Current rule to apply
	fuzzer_stem *Next;			// Next stem in rCost order
	fuzzer_stem *Hash;			// Next stem with same hash on zBasis
	fuzzer_cost BaseCost;		// Base cost of getting to zBasis
	fuzzer_cost CostX;			// Precomputed rBaseCost + pRule->rCost
	fuzzer_len BasisLength;		// Length of the zBasis string
	fuzzer_len N;				// Apply pRule at this character offset
};

// A fuzzer virtual-table object 
struct fuzzer_vtab
{
	IVTable base;				// Base class - must be first
	char *ClassName;			// Name of this class.  Default: "fuzzer"
	fuzzer_rule *Rule;			// All active rules in this fuzzer
	int Cursors;				// Number of active cursors
};

#define FUZZER_HASH  4001		// Hash table size
#define FUZZER_NQUEUE  20		// Number of slots on the stem queue

// A fuzzer cursor object
struct fuzzer_cursor
{
	IVTableCursor base;			// Base class - must be first
	int64 Rowid;				// The rowid of the current word
	fuzzer_vtab *Vtab;			// The virtual table this cursor belongs to
	fuzzer_cost Limit;			// Maximum cost of any term
	fuzzer_stem *Stem;			// Stem with smallest rCostX
	fuzzer_stem *Done;			// Stems already processed to completion
	fuzzer_stem *Queues[FUZZER_NQUEUE];  // Queue of stems with higher rCostX
	int MaxQueue;				// Largest used index in aQueue[]
	char *Buf;					// Temporary use buffer
	int BufLength;					// Bytes allocated for zBuf
	int Stems;					// Number of stems allocated
	int Ruleset;				// Only process rules from this ruleset
	fuzzer_rule NullRule;		// Null rule used first
	fuzzer_stem *Hashs[FUZZER_HASH]; // Hash of previously generated terms
};

// The two input rule lists are both sorted in order of increasing cost.  Merge them together into a single list, sorted by cost, and
// return a pointer to the head of that list.
__device__ static fuzzer_rule *fuzzerMergeRules(fuzzer_rule *a, fuzzer_rule *b)
{
	fuzzer_rule head;
	fuzzer_rule *tail =  &head;
	while (a && b)
	{
		if (a->Cost <= b->Cost) { tail->Next = a; tail = a; a = a->Next; }
		else { tail->Next = b; tail = b; b = b->Next; }
	}
	tail->Next = (!a ? b : a);
	return head.Next;
}

// Statement pStmt currently points to a row in the fuzzer data table. This function allocates and populates a fuzzer_rule structure according to
// the content of the row.
//
// If successful, *ppRule is set to point to the new object and SQLITE_OK is returned. Otherwise, *ppRule is zeroed, *pzErr may be set to point
// to an error message and an SQLite error code returned.
__device__ static RC fuzzerLoadOneRule(fuzzer_vtab *p, Vdbe *stmt, fuzzer_rule **rule, char **err)
{
	int64 ruleset = Vdbe::Column_Int64(stmt, 0);
	const char *from = (const char *)Vdbe::Column_Text(stmt, 1);
	const char *to = (const char *)Vdbe::Column_Text(stmt, 2);
	int cost = Vdbe::Column_Int(stmt, 3);
	if (!from) from = "";
	if (!to) to = "";
	int fromLength = (int)_strlen(from); // Size of string zFrom, in bytes
	int toLength = (int)_strlen(to);// Size of string zTo, in bytes

	// Silently ignore null transformations
	if (!_strcmp(from, to))
	{
		*rule = nullptr;
		return RC_OK;
	}

	RC rc = RC_OK;
	fuzzer_rule *newRule = nullptr; // New rule object to return
	if (cost <= 0 || cost > FUZZER_MX_COST) { *err = _mprintf("%s: cost must be between 1 and %d", p->ClassName, FUZZER_MX_COST); rc = RC_ERROR; }
	else if (fromLength > FUZZER_MX_LENGTH || toLength > FUZZER_MX_LENGTH) { *err = _mprintf("%s: maximum string length is %d", p->ClassName, FUZZER_MX_LENGTH); rc = RC_ERROR; }
	else if (ruleset < 0 || ruleset > FUZZER_MX_RULEID) { *err = _mprintf("%s: ruleset must be between 0 and %d", p->ClassName, FUZZER_MX_RULEID); rc = RC_ERROR; }
	else
	{
		newRule = (fuzzer_rule *)_alloc( sizeof(*rule) + fromLength + toLength);
		if (!newRule)
			rc = RC_NOMEM;
		else
		{
			memset(newRule, 0, sizeof(*newRule));
			newRule->From = &newRule->To[toLength+1];
			newRule->FromLength = fromLength;
			memcpy(newRule->From, from, fromLength+1);
			memcpy(newRule->To, to, toLength+1);
			newRule->ToLength = toLength;
			newRule->Cost = cost;
			newRule->Ruleset = (int)ruleset;
		}
	}
	*rule = newRule;
	return rc;
}

// Load the content of the fuzzer data table into memory.
__device__ static RC fuzzerLoadRules(Context *ctx, fuzzer_vtab *p, const char *dbName, const char *dataName, char **err)
{
	RC rc = RC_OK;
	fuzzer_rule *head = nullptr;
	char *sql = _mprintf("SELECT * FROM %Q.%Q", dbName, dataName); // SELECT used to read from rules table
	if (!sql)
		rc = RC_NOMEM;
	else
	{
		Vdbe *stmt = nullptr;
		rc = Prepare::Prepare_v2(ctx, sql, -1, &stmt, nullptr);
		if (rc != RC_OK) { *err = _mprintf("%s: %s", p->ClassName, Main::ErrMsg(ctx)); }
		else if (Vdbe::Column_Count(stmt) != 4) { *err = _mprintf("%s: %s has %d columns, expected 4", p->ClassName, dataName, Vdbe::Column_Count(stmt)); rc = RC_ERROR; }
		else
		{
			while (rc == RC_OK && stmt->Step() == RC_ROW)
			{
				fuzzer_rule *rule = nullptr;
				rc = fuzzerLoadOneRule(p, stmt, &rule, err);
				if (rule)
				{
					rule->Next = head;
					head = rule;
				}
			}
		}
		RC rc2 = Vdbe::Finalize(stmt); // finalize() return code
		if (rc == RC_OK ) rc = rc2;
	}
	_free(sql);

	// All rules are now in a singly linked list starting at pHead. This block sorts them by cost and then sets fuzzer_vtab.pRule to point to 
	// point to the head of the sorted list.
	if (rc == RC_OK)
	{
		unsigned int i;
		fuzzer_rule *x;
		fuzzer_rule *as[15];
		for (i = 0; i < _lengthof(as); i++) as[i] = 0;
		while ((x = head) != nullptr)
		{
			head = x->Next;
			x->Next = nullptr;
			for (i = 0; as[i] && i < _lengthof(as)-1; i++)
			{
				x = fuzzerMergeRules(as[i], x);
				as[i] = nullptr;
			}
			as[i] = fuzzerMergeRules(as[i], x);
		}
		for (x = as[0], i = 1; i < _lengthof(as); i++)
			x = fuzzerMergeRules(as[i], x);
		p->Rule = fuzzerMergeRules(p->Rule, x);
	}
	else
	{
		// An error has occurred. Setting p->pRule to point to the head of the allocated list ensures that the list will be cleaned up in this case.
		_assert(!p->Rule);
		p->Rule = head;
	}
	return rc;
}

// This function converts an SQL quoted string into an unquoted string and returns a pointer to a buffer allocated using sqlite3_malloc() 
// containing the result. The caller should eventually free this buffer using sqlite3_free.
//
// Examples:
//     "abc"   becomes   abc
//     'xyz'   becomes   xyz
//     [pqr]   becomes   pqr
//     `mno`   becomes   mno
__device__ static char *fuzzerDequote(const char *in)
{
	int inLength = (int)_strlen(in); // Size of input string, in bytes
	char *out = (char *)_alloc(inLength+1); // Output (dequoted) string
	if (out)
	{
		char q = in[0]; // Quote character (if any )
		if (q != '[' && q != '\'' && q != '"' && q != '`')
			memcpy(out, in, inLength+1);
		else
		{
			int outIdx = 0; // Index of next byte to write to output
			if (q == '[' ) q = ']';
			for (int inIdx = 1; inIdx < inLength; inIdx++) // Index of next byte to read from input
			{
				if (in[inIdx] == q) inIdx++;
				out[outIdx++] = in[inIdx];
			}
		}
		_assert((int)_strlen(out) <= inLength);
	}
	return out;
}

// xDisconnect/xDestroy method for the fuzzer module.
__device__ static RC fuzzerDisconnect(IVTable *vtab)
{
	fuzzer_vtab *p = (fuzzer_vtab *)vtab;
	_assert(p->Cursors == 0);
	while (p->Rule)
	{
		fuzzer_rule *rule = p->Rule;
		p->Rule = rule->Next;
		_free(rule);
	}
	_free(p);
	return RC_OK;
}

// xConnect/xCreate method for the fuzzer module. Arguments are:
//   argv[0]   -> module name ("fuzzer")
//   argv[1]   -> database name
//   argv[2]   -> table name
//   argv[3]   -> fuzzer rule table name
__device__ static RC fuzzerConnect(Context *ctx, void *aux, int argc, const char *const args[], IVTable **vtab, char **err)
{
	const char *module = args[0];
	const char *dbName = args[1];
	RC rc = RC_OK;
	fuzzer_vtab *newVtab = nullptr; // New virtual table
	if (argc != 4) { *err = _mprintf("%s: wrong number of CREATE VIRTUAL TABLE arguments", module); rc = RC_ERROR; }
	else
	{
		int moduleLength = (int)_strlen(module); // Length of zModule, in bytes
		newVtab = (fuzzer_vtab *)_alloc(sizeof(*newVtab) + moduleLength + 1);
		if (!newVtab)
			rc = RC_NOMEM;
		else
		{
			memset(newVtab, 0, sizeof(*newVtab));
			newVtab->ClassName = (char *)&newVtab[1];
			memcpy(newVtab->ClassName, module, moduleLength+1);
			char *tableName = fuzzerDequote(args[3]); // Dequoted name of fuzzer data table
			if (!tableName)
				rc = RC_NOMEM;
			else
			{
				rc = fuzzerLoadRules(ctx, newVtab, dbName, tableName, err);
				_free(tableName);
			}
			if (rc == RC_OK)
				rc = VTable::DeclareVTable(ctx, "CREATE TABLE x(word,distance,ruleset)");
			if (rc != RC_OK)
			{
				fuzzerDisconnect((IVTable *)newVtab);
				newVtab = nullptr;
			}
		}
	}
	*vtab = (IVTable *)newVtab;
	return rc;
}

// Open a new fuzzer cursor.
__device__ static RC fuzzerOpen(IVTable *vtab, IVTableCursor **cursor)
{
	fuzzer_vtab *p = (fuzzer_vtab *)vtab;
	fuzzer_cursor *cur = (fuzzer_cursor *)_alloc(sizeof(*cur));
	if (!cur) return RC_NOMEM;
	memset(cur, 0, sizeof(*cur));
	cur->Vtab = p;
	*cursor = &cur->base;
	p->Cursors++;
	return RC_OK;
}

// Free all stems in a list.
__device__ static void fuzzerClearStemList(fuzzer_stem *stem)
{
	while (stem)
	{
		fuzzer_stem *next = stem->Next;
		_free(stem);
		stem = next;
	}
}

// Free up all the memory allocated by a cursor.  Set it rLimit to 0 to indicate that it is at EOF.
__device__ static void fuzzerClearCursor(fuzzer_cursor *cur, bool clearHash)
{
	fuzzerClearStemList(cur->Stem);
	fuzzerClearStemList(cur->Done);
	for (int i = 0; i < FUZZER_NQUEUE; i++) fuzzerClearStemList(cur->Queues[i]);
	cur->Limit = (fuzzer_cost)0;
	if (clearHash && cur->Stems)
	{
		cur->MaxQueue = 0;
		cur->Stem = nullptr;
		cur->Done = nullptr;
		memset(cur->Queues, 0, sizeof(cur->Queues));
		memset(cur->Hashs, 0, sizeof(cur->Hashs));
	}
	cur->Stems = 0;
}

// Close a fuzzer cursor.
__device__ static RC fuzzerClose(IVTableCursor *cur)
{
	fuzzer_cursor *cur2 = (fuzzer_cursor *)cur;
	fuzzerClearCursor(cur2, 0);
	_free(cur2->Buf);
	cur2->Vtab->Cursors--;
	_free(cur2);
	return RC_OK;
}

// Compute the current output term for a fuzzer_stem.
__device__ static RC fuzzerRender(fuzzer_stem *stem, char **buf, int *bufLength)
{
	const fuzzer_rule *rule = stem->Rule;
	int n = stem->BasisLength + rule->ToLength - rule->FromLength; // Size of output term without nul-term
	if ((*bufLength) < n+1)
	{
		*buf = (char *)_realloc(*buf, n+100);
		if (!*buf) return RC_NOMEM;
		*bufLength = n+100;
	}
	n = stem->N;
	char *z = *buf; // Buffer to assemble output term in
	if (n < 0)
		memcpy(z, stem->Basis, stem->BasisLength+1);
	else
	{
		memcpy(z, stem->Basis, n);
		memcpy(&z[n], rule->To, rule->ToLength);
		memcpy(&z[n+rule->ToLength], &stem->Basis[n+rule->FromLength], stem->BasisLength-n-rule->FromLength+1);
	}
	_assert(z[stem->BasisLength + rule->ToLength - rule->FromLength] == 0);
	return RC_OK;
}

// Compute a hash on zBasis.
__device__ static unsigned int fuzzerHash(const char *z)
{
	unsigned int h = 0;
	while (*z) { h = (h<<3) ^ (h>>29) ^ *(z++); }
	return h % FUZZER_HASH;
}

// Current cost of a stem
__device__ static fuzzer_cost fuzzerCost(fuzzer_stem *stem)
{
	return stem->CostX = stem->BaseCost + stem->Rule->Cost;
}

#if 0
// Print a description of a fuzzer_stem on stderr.
__device__ static void fuzzerStemPrint(const char *prefix, fuzzer_stem *stem, const char *suffix)
{
	if (stem->N < 0)
		fprintf(stderr, "%s[%s](%d)-->self%s", prefix, stem->Basis, stem->BaseCost, suffix);
	else
	{
		char *buf = nullptr;
		int bufLength = 0;
		if (fuzzerRender(stem, &buf, &bufLength) != RC_OK) return;
		fprintf(stderr, "%s[%s](%d)-->{%s}(%d)%s", prefix, stem->Basis, stem->BaseCost, buf, stem->N, suffix);
		_free(buf);
	}
}
#endif

// Return 1 if the string to which the cursor is point has already been emitted.  Return 0 if not.  Return -1 on a memory allocation failures.
__device__ static int fuzzerSeen(fuzzer_cursor *cur, fuzzer_stem *stem)
{
	if (fuzzerRender(stem, &cur->Buf, &cur->BufLength) == RC_NOMEM)
		return -1;
	unsigned int h = fuzzerHash(cur->Buf);
	fuzzer_stem *lookup = cur->Hashs[h];
	while (lookup && _strcmp(lookup->Basis, cur->Buf))
		lookup = lookup->Hash;
	return (lookup != 0);
}

// If argument pRule is NULL, this function returns false.
// Otherwise, it returns true if rule pRule should be skipped. A rule should be skipped if it does not belong to rule-set iRuleset, or if
// applying it to stem pStem would create a string longer than FUZZER_MX_OUTPUT_LENGTH bytes.
__device__ static int fuzzerSkipRule(const fuzzer_rule *rule, fuzzer_stem *stem, int ruleset)
{
	return rule && (rule->Ruleset != ruleset || (stem->BasisLength + rule->ToLength - rule->FromLength) > FUZZER_MX_OUTPUT_LENGTH);
}

// Advance a fuzzer_stem to its next value.   Return 0 if there are no more values that can be generated by this fuzzer_stem.  Return
// -1 on a memory allocation failure.
__device__ static int fuzzerAdvance(fuzzer_cursor *cur, fuzzer_stem *stem)
{
	const fuzzer_rule *rule;
	while ((rule = stem->Rule) != 0)
	{
		_assert(rule == &cur->NullRule || rule->Ruleset == cur->Ruleset);
		while (stem->N < stem->BasisLength - rule->FromLength)
		{
			stem->N++;
			if (rule->FromLength == 0 || !memcmp(&stem->Basis[stem->N], rule->From, rule->FromLength))
			{
				// Found a rewrite case.  Make sure it is not a duplicate
				int rc = fuzzerSeen(cur, stem);
				if (rc < 0) return -1;
				if (rc == 0) { fuzzerCost(stem); return 1; }
			}
		}
		stem->N = -1;
		do
		{
			rule = rule->Next;
		} while (fuzzerSkipRule(rule, stem, cur->Ruleset));
		stem->Rule = rule;
		if (rule && fuzzerCost(stem) > cur->Limit) stem->Rule = 0;
	}
	return 0;
}

// The two input stem lists are both sorted in order of increasing rCostX.  Merge them together into a single list, sorted by rCostX, and
// return a pointer to the head of that new list.
__device__ static fuzzer_stem *fuzzerMergeStems(fuzzer_stem *a, fuzzer_stem *b)
{
	fuzzer_stem head;
	fuzzer_stem *tail = &head;
	while (a && b)
	{
		if (a->CostX <= b->CostX) { tail->Next = a; tail = a; a = a->Next; }
		else { tail->Next = b; tail = b; b = b->Next; }
	}
	tail->Next = (!a ? b : a);
	return head.Next;
}

// Load pCur->pStem with the lowest-cost stem.  Return a pointer to the lowest-cost stem.
__device__ static fuzzer_stem *fuzzerLowestCostStem(fuzzer_cursor *cur)
{
	if (!cur->Stem)
	{
		int bestId = -1;
		fuzzer_stem *best = nullptr;
		for (int i = 0; i <= cur->MaxQueue; i++)
		{
			fuzzer_stem *x = cur->Queues[i];
			if (!x) continue;
			if (!best || best->CostX > x->CostX) { best = x; bestId = i; }
		}
		if (best)
		{
			cur->Queues[bestId] = best->Next;
			best->Next = nullptr;
			cur->Stem = best;
		}
	}
	return cur->Stem;
}

// Insert pNew into queue of pending stems.  Then find the stem with the lowest rCostX and move it into pCur->pStem.
// list.  The insert is done such the pNew is in the correct order according to fuzzer_stem.zBaseCost+fuzzer_stem.pRule->rCost.
__device__ static fuzzer_stem *fuzzerInsert(fuzzer_cursor *cur, fuzzer_stem *newStem)
{
	// If pCur->pStem exists and is greater than pNew, then make pNew the new pCur->pStem and insert the old pCur->pStem instead.
	fuzzer_stem *x;
	if ((x = cur->Stem) != 0 && x->CostX > newStem->CostX)
	{
		newStem->Next = nullptr;
		cur->Stem = newStem;
		newStem = x;
	}
	// Insert the new value
	newStem->Next = nullptr;
	x = newStem;
	int i;
	for (i = 0; i <= cur->MaxQueue; i++)
	{
		if (cur->Queues[i])
		{
			x = fuzzerMergeStems(x, cur->Queues[i]);
			cur->Queues[i] = nullptr;
		}
		else
		{
			cur->Queues[i] = x;
			break;
		}
	}
	if (i > cur->MaxQueue)
	{
		if (i < FUZZER_NQUEUE)
		{
			cur->MaxQueue = i;
			cur->Queues[i] = x;
		}
		else
		{
			_assert(cur->MaxQueue == FUZZER_NQUEUE-1);
			x = fuzzerMergeStems(x, cur->Queues[FUZZER_NQUEUE-1]);
			cur->Queues[FUZZER_NQUEUE-1] = x;
		}
	}
	return fuzzerLowestCostStem(cur);
}

// Allocate a new fuzzer_stem.  Add it to the hash table but do not link it into either the pCur->pStem or pCur->pDone lists.
__device__ static fuzzer_stem *fuzzerNewStem(fuzzer_cursor *cur, const char *word, fuzzer_cost baseCost)
{
	fuzzer_stem *newStem = (fuzzer_stem *)_alloc(sizeof(*newStem) + (int)_strlen(word) + 1);
	if (!newStem) return nullptr;
	_memset(newStem, 0, sizeof(*newStem));
	newStem->Basis = (char *)&newStem[1];
	newStem->BasisLength = (int)_strlen(word);
	_memcpy(newStem->Basis, word, newStem->BasisLength+1);
	fuzzer_rule *rule = cur->Vtab->Rule;
	while (fuzzerSkipRule(rule, newStem, cur->Ruleset))
		rule = rule->Next;
	newStem->Rule = rule;
	newStem->N = -1;
	newStem->BaseCost = newStem->CostX = baseCost;
	unsigned int h = fuzzerHash(newStem->Basis);
	newStem->Hash = cur->Hashs[h];
	cur->Hashs[h] = newStem;
	cur->Stems++;
	return newStem;
}

// Advance a cursor to its next row of output
__device__ static RC fuzzerNext(IVTableCursor *cur_)
{
	fuzzer_cursor *cur = (fuzzer_cursor*)cur_;
	cur->Rowid++;
	RC rc;
	// Use the element the cursor is currently point to to create a new stem and insert the new stem into the priority queue.
	fuzzer_stem *stem = cur->Stem;
	if (stem->CostX > 0)
	{
		rc = fuzzerRender(stem, &cur->Buf, &cur->BufLength);
		if (rc == RC_NOMEM) return RC_NOMEM;
		fuzzer_stem *newStem = fuzzerNewStem(cur, cur->Buf, stem->CostX);
		if (newStem)
		{
			if (!fuzzerAdvance(cur, newStem)) { newStem->Next = cur->Done; cur->Done = newStem; }
			else if (fuzzerInsert(cur, newStem) == newStem) return RC_OK;
		}
		else return RC_NOMEM;
	}
	// Adjust the priority queue so that the first element of the stem list is the next lowest cost word.
	while ((stem = cur->Stem) != 0)
	{
		int res = fuzzerAdvance(cur, stem);
		if (res < 0)
			return RC_NOMEM;
		else if (res > 0)
		{
			cur->Stem = 0;
			stem = fuzzerInsert(cur, stem);
			if ((rc = (RC)fuzzerSeen(cur, stem)) != 0)
			{
				if (rc < 0) return RC_NOMEM;
				continue;
			}
			return RC_OK; // New word found
		}
		cur->Stem = nullptr;
		stem->Next = cur->Done;
		cur->Done = stem;
		if (fuzzerLowestCostStem(cur))
		{
			rc = (RC)fuzzerSeen(cur, cur->Stem);
			if (rc < 0) return RC_NOMEM;
			if (rc == 0) return RC_OK;
		}
	}
	// Reach this point only if queue has been exhausted and there is nothing left to be output.
	cur->Limit = (fuzzer_cost)0;
	return RC_OK;
}

// Called to "rewind" a cursor back to the beginning so that it starts its output over again.  Always called at least once
// prior to any fuzzerColumn, fuzzerRowid, or fuzzerEof call.
__device__ static RC fuzzerFilter(IVTableCursor *cur_, int idxNum, const char *idxStr, int argc, Mem **args)
{
	fuzzer_cursor *cur = (fuzzer_cursor *)cur_;
	fuzzer_stem *stem;
	fuzzerClearCursor(cur, 1);
	const char *word = "";
	cur->Limit = 2147483647;
	int idx = 0;
	if (idxNum & 1)
	{
		word = (const char*)Vdbe::Value_Text(args[0]);
		idx++;
	}
	if (idxNum & 2)
	{
		cur->Limit = (fuzzer_cost)Vdbe::Value_Int(args[idx]);
		idx++;
	}
	if (idxNum & 4)
	{
		cur->Ruleset = (fuzzer_cost)Vdbe::Value_Int(args[idx]);
		idx++;
	}
	cur->NullRule.Next = cur->Vtab->Rule;
	cur->NullRule.Cost = 0;
	cur->NullRule.FromLength = 0;
	cur->NullRule.ToLength = 0;
	cur->NullRule.From = "";
	cur->Rowid = 1;
	_assert(cur->Stem == nullptr);
	// If the query term is longer than FUZZER_MX_OUTPUT_LENGTH bytes, this query will return zero rows.
	if ((int)_strlen(word) < FUZZER_MX_OUTPUT_LENGTH)
	{
		cur->Stem = stem = fuzzerNewStem(cur, word, (fuzzer_cost)0);
		if (!stem) return RC_NOMEM;
		stem->Rule = &cur->NullRule;
		stem->N = stem->BasisLength;
	}
	else
		cur->Limit = 0;
	return RC_OK;
}

// Only the word and distance columns have values.  All other columns return NULL
__device__ static RC fuzzerColumn(IVTableCursor *cur_, FuncContext *fctx, int i)
{
	fuzzer_cursor *cur = (fuzzer_cursor *)cur_;
	if (i == 0)
	{
		// the "word" column
		if (fuzzerRender(cur->Stem, &cur->Buf, &cur->BufLength) == RC_NOMEM)
			return RC_NOMEM;
		Vdbe::Result_Text(fctx, cur->Buf, -1, DESTRUCTOR_TRANSIENT);
	}
	else if (i == 1)
		// the "distance" column
		Vdbe::Result_Int(fctx, cur->Stem->CostX);
	else
		// All other columns are NULL
		Vdbe::Result_Null(fctx);
	return RC_OK;
}

// The rowid.
__device__ static RC fuzzerRowid(IVTableCursor *cur_, int64 *rowid)
{
	fuzzer_cursor *cur = (fuzzer_cursor *)cur_;
	*rowid = cur->Rowid;
	return RC_OK;
}

// When the fuzzer_cursor.rLimit value is 0 or less, that is a signal that the cursor has nothing more to output.
__device__ static bool fuzzerEof(IVTableCursor *cur_)
{
	fuzzer_cursor *cur = (fuzzer_cursor *)cur_;
	return cur->Limit <= (fuzzer_cost)0;
}

// Search for terms of these forms:
//   (A)    word MATCH $str
//   (B1)   distance < $value
//   (B2)   distance <= $value
//   (C)    ruleid == $ruleid
//
// The distance< and distance<= are both treated as distance <=. The query plan number is a bit vector:
//   bit 1:   Term of the form (A) found
//   bit 2:   Term like (B1) or (B2) found
//   bit 3:   Term like (C) found
//
// If bit-1 is set, $str is always in filter.argv[0].  If bit-2 is set then $value is in filter.argv[0] if bit-1 is clear and is in 
// filter.argv[1] if bit-1 is set.  If bit-3 is set, then $ruleid is in filter.argv[0] if bit-1 and bit-2 are both zero, is in
// filter.argv[1] if exactly one of bit-1 and bit-2 are set, and is in filter.argv[2] if both bit-1 and bit-2 are set.
__device__ static RC fuzzerBestIndex(IVTable *tab, IIndexInfo *idxInfo)
{
	int plan = 0;
	int distTerm = -1;
	int rulesetTerm = -1;
	const IIndexInfo::Constraint *constraint = idxInfo->Constraints.data;
	for (int i = 0; i < idxInfo->Constraints.length; i++, constraint++)
	{
		if (!constraint->Usable) continue;
		if ((plan&1) == 0 && constraint->Column == 0 && constraint->OP == INDEX_CONSTRAINT_MATCH)
		{
			plan |= 1;
			idxInfo->ConstraintUsages[i].ArgvIndex = 1;
			idxInfo->ConstraintUsages[i].Omit = 1;
		}
		if ((plan&2) == 0 && constraint->Column == 1 && (constraint->OP == INDEX_CONSTRAINT_LT || constraint->OP == INDEX_CONSTRAINT_LE))
		{
			plan |= 2;
			distTerm = i;
		}
		if ((plan&4) == 0 && constraint->Column == 2 && constraint->OP == INDEX_CONSTRAINT_EQ)
		{
			plan |= 4;
			idxInfo->ConstraintUsages[i].Omit = 1;
			rulesetTerm = i;
		}
	}
	if (plan&2)
		idxInfo->ConstraintUsages[distTerm].ArgvIndex = 1+((plan&1) != 0);
	if (plan&4)
	{
		int idx = 1;
		if (plan&1) idx++;
		if (plan&2) idx++;
		idxInfo->ConstraintUsages[rulesetTerm].ArgvIndex = idx;
	}
	idxInfo->IdxNum = plan;
	if (idxInfo->OrderBys.length == 1 && idxInfo->OrderBys[0].Column == 1 && !idxInfo->OrderBys[0].Desc)
		idxInfo->OrderByConsumed = true;
	idxInfo->EstimatedCost = (double)10000;
	return RC_OK;
}

// A virtual table module that implements the "fuzzer".
__constant__ static ITableModule _fuzzerModule = {
	0,                  // Version
	fuzzerConnect,
	fuzzerConnect,
	fuzzerBestIndex,
	fuzzerDisconnect, 
	fuzzerDisconnect,
	fuzzerOpen,			// Open - open a cursor
	fuzzerClose,		// Close - close a cursor
	fuzzerFilter,		// Filter - configure scan constraints
	fuzzerNext,			// Next - advance a cursor
	fuzzerEof,			// Eof - check for end of scan
	fuzzerColumn,		// Column - read data
	fuzzerRowid,		// Rowid - read data
	nullptr,			// xUpdate
	nullptr,			// xBegin
	nullptr,			// xSync
	nullptr,			// xCommit
	nullptr,			// xRollback
	nullptr,			// xFindMethod
	nullptr,			// xRename
};

#endif

// Register the fuzzer virtual table
__device__ RC fuzzer_register(Context *ctx)
{
	RC rc = RC_OK;
#ifndef OMIT_VIRTUALTABLE
	rc = VTable::CreateModule(ctx, "fuzzer", &_fuzzerModule, nullptr, nullptr);
#endif
	return rc;
}

#ifdef _TEST
#include <Jim.h>

// Decode a pointer to an sqlite3 object.
__device__ extern int GetDbPointer(Jim_Interp *interp, const char *a, Context **ctx);

// Register the echo virtual table module.
__device__ static int register_fuzzer_module(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "DB");
		return JIM_ERROR;
	}
	Context *ctx;
	if (GetDbPointer(interp, Jim_String(args[1]), &ctx)) return JIM_ERROR;
	fuzzer_register(ctx);
	return JIM_OK;
}

// Register commands with the TCL interpreter.
__constant__ static struct {
	char *Name;
	Jim_CmdProc *Proc;
	ClientData ClientData;
} _objCmds[] = {
	{ "register_fuzzer_module", register_fuzzer_module, nullptr },
};
__device__ int Sqlitetestfuzzer_Init(Jim_Interp *interp)
{
	for (int i = 0; i < _lengthof(_objCmds); i++)
		Jim_CreateCommand(interp, _objCmds[i].Name, _objCmds[i].Proc, _objCmds[i].ClientData, nullptr);
	return JIM_OK;
}

#endif
