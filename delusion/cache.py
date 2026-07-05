from diskcache import Cache as DiskCache

import delusion

CHAT_CACHE: DiskCache = DiskCache(
    directory=delusion.dirs.user_cache_path.joinpath("chat"),
    size_limit=(512 * 1024**2), # MB
)
