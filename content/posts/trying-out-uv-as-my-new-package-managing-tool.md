+++
title = "Trying Out uv as my new package managing tool"
date = 2025-07-20T00:06:57+09:00
draft = true
categories = ['']
+++

## Intro

Dependency hell is a known problem in Machine Learning ecosystem. Hardware(eg. NVIDIA chips) with major libraries such as PyTorch, NumPy etc. can easily create all sorts of dependency issues. That is why making a system for maximum reproducability is important.

I've previously used poetry or conda for package managing for Python but found it hard and clunky to use sometimes. Recently I've found a rising tool for package managing called [uv](https://github.com/astral-sh/uv/). This is a brief introduction of what uv is and my first impression of the tool.



## Performance

I heard uv was very fast, optimized by using Rust, so I wanted to test how fast it was compared to pip install. 

**uv**: ~2m 35s which is not so bad

```bash
❯ uv add torch  
Resolved 31 packages in 381ms  
Prepared 25 packages in 2m 35s  
Installed 25 packages in 240ms  
+ filelock==3.18.0  
...
+ typing-extensions==4.14.1
```

**pip** : **~9m 34s** ngl it took forever to install them 

```bash
❯ time pip install torch  
Collecting torch  
...
noglob pip install torch 65.50s user 23.14s system 15% cpu 9:34.48 total
```

Conclusion: uv is much faster and efficient when importing libraries.


## Resources
https://github.com/astral-sh/uv/
https://docs.astral.sh/uv/



When ready to publish:
1. Add appropriate category: `categories = ['Thoughts']` or `['ML']`
2. Change `draft = false` in frontmatter
3. Git plugin will auto-commit and deploy