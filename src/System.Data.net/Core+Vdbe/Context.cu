#include "Core+Vdbe.cu.h"

namespace CORE_NAME
{
	__device__ RC Context::Status(CTXSTATUS op, int *current, int *highwater, bool resetFlag)
	{
		RC rc = RC_OK;
		_mutex_enter(Mutex);
		switch (op)
		{
		case CTXSTATUS_LOOKASIDE_USED: {
			*current = Lookaside.Outs;
			*highwater = Lookaside.MaxOuts;
			if (resetFlag)
				Lookaside.MaxOuts = Lookaside.Outs;
			break; }
		case CTXSTATUS_LOOKASIDE_HIT:
		case CTXSTATUS_LOOKASIDE_MISS_SIZE:
		case CTXSTATUS_LOOKASIDE_MISS_FULL: {
			ASSERTCOVERAGE(op == CTXSTATUS_LOOKASIDE_HIT);
			ASSERTCOVERAGE(op == CTXSTATUS_LOOKASIDE_MISS_SIZE);
			ASSERTCOVERAGE(op == CTXSTATUS_LOOKASIDE_MISS_FULL);
			_assert((op - CTXSTATUS_LOOKASIDE_HIT) >= 0);
			_assert((op - CTXSTATUS_LOOKASIDE_HIT) < 3);
			//
			*current = 0;
			*highwater = Lookaside.Stats[op - CTXSTATUS_LOOKASIDE_HIT];
			if (resetFlag)
				Lookaside.Stats[op - CTXSTATUS_LOOKASIDE_HIT] = 0;
			break; }
		case CTXSTATUS_CACHE_USED: {
			// Return an approximation for the amount of memory currently used by all pagers associated with the given database connection.  The
			// highwater mark is meaningless and is returned as zero.
			int totalUsed = 0;
			int i;
			Btree::EnterAll(this);
			for (i = 0; i < DBs.length; i++)
			{
				Btree *bt = DBs[i].Bt;
				if (bt)
				{
					Pager *pager = bt->get_Pager();
					totalUsed += pager->get_MemUsed();
				}
			}
			Btree::LeaveAll(this);
			//
			*current = totalUsed;
			*highwater = 0;
			break; }
		case CTXSTATUS_SCHEMA_USED: {
			// *pCurrent gets an accurate estimate of the amount of memory used to store the schema for all databases (main, temp, and any ATTACHed
			// databases.  *pHighwater is set to zero.
			int i;                      // Used to iterate through schemas
			int bytes = 0;              // Used to accumulate return value

			Btree::EnterAll(this);
			BytesFreed = &bytes;
			for (i = 0; i < DBs.length; i++)
			{
				Schema *schema = DBs[i].Schema;
				if (_ALWAYS(schema != nullptr))
				{
					bytes += _ROUND8(sizeof(HashElem)) * (schema->TableHash.Count + schema->TriggerHash.Count + schema->IndexHash.Count + schema->FKeyHash.Count);
					bytes += (int)_allocsize(schema->TableHash.Table);
					bytes += (int)_allocsize(schema->TriggerHash.Table);
					bytes += (int)_allocsize(schema->IndexHash.Table);
					bytes += (int)_allocsize(schema->FKeyHash.Table);
					HashElem *p;
					for (p = schema->TriggerHash.First; p; p = p->Next) Trigger::DeleteTrigger(this, (Trigger *)p->Data);
					for (p = schema->TableHash.First; p; p = p->Next) Parse::DeleteTable(this, (Table *)p->Data);
				}
			}
			BytesFreed = 0;
			Btree::LeaveAll(this);
			//
			*highwater = 0;
			*current = bytes;
			break; }
		case CTXSTATUS_STMT_USED: {
			// *pCurrent gets an accurate estimate of the amount of memory used to store all prepared statements. *pHighwater is set to zero.
			int bytes = 0; // Used to accumulate return value
			BytesFreed = &bytes;
			for (Vdbe *v = Vdbes; v; v = v->Next)
			{
				v->ClearObject(this);
				_tagfree(this, v);
			}
			BytesFreed = 0;
			//
			*highwater = 0;
			*current = bytes;
			break; }
		case CTXSTATUS_CACHE_HIT:
		case CTXSTATUS_CACHE_MISS:
		case CTXSTATUS_CACHE_WRITE:{
			// Set *pCurrent to the total cache hits or misses encountered by all pagers the database handle is connected to. *pHighwater is always set  to zero.
			_assert(CTXSTATUS_CACHE_MISS == CTXSTATUS_CACHE_HIT+1);
			_assert(CTXSTATUS_CACHE_WRITE == CTXSTATUS_CACHE_HIT+2);
			int r = 0;
			for (int i = 0; i < DBs.length; i++)
			{
				if (DBs[i].Bt)
				{
					Pager *pager = DBs[i].Bt->get_Pager();
					pager->CacheStat(op, resetFlag, &r);
				}
			}
			//
			*highwater = 0;
			*current = r;
			break; }
		default: {
			rc = RC_ERROR; }
		}
		_mutex_leave(Mutex);
		return rc;
	}	
}