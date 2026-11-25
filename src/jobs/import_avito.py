from collectors.registry import source_adapter
from jobs.catalog_freshness import RECHECK_AFTER


async def import_avito_pages(path, sessions, *, refresh_after=RECHECK_AFTER):
    return await source_adapter("avito").import_pages(path, sessions, refresh_after=refresh_after)
