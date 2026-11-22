from collectors.registry import source_adapter


async def import_avito_pages(path, sessions):
    return await source_adapter("avito").import_pages(path, sessions)
