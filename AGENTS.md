# AGENTS.md
- Keep the abstraction close, and always ask yourself whether your design is leaky (refer to [leaky abstraction](https://en.wikipedia.org/wiki/Leaky_abstraction)).
- For all of the config, to keep it simple, only using the `ServerArgs` from the `python/zookeeper/server_args.py`.
- Write no comments in the code, if needed, the user will ask. Keep the comments from the user and just put that as a side-rant, not a direction to follow. Ranting is ranting.
- No hard-coded config.
- all must be written in mlx, no torch.