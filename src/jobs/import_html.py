from collectors.registry import source_adapter


async def import_html_snapshots(directory, repository, *, base_url):
    return await source_adapter("html").import_snapshots(directory, repository, base_url=base_url)
