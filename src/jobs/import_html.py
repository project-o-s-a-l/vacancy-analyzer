from collectors.registry import source_adapter


async def resume_html_import(directory, sessions, *, base_url):
    return await source_adapter("html").import_pages(directory, sessions, base_url=base_url)


async def prune_snapshots(directory, sessions, *, days, apply=False):
    return await source_adapter("html").prune(directory, sessions, days=days, apply=apply)
