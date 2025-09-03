+++
title = "limitations-of-current-rl"
date = 2025-09-03T11:13:40+09:00
draft = false
categories = ['Thoughts']
tags = ['short']
+++

Just read the paper _Welcome to the Era of Experience_ by Richard S. Sutton and David Silver, and while I admit the potential impact RL will have, I’m pretty concerned about what these authors believe or are trying to create.

I believe AI in general needs to be controlled and understood by humans as much as possible, especially for making important and impactful judgments. However, RL lacks this understandability and controllability because of its unexplainable, black-box decision mechanism. It simply makes choices that will gain the maximum reward based on the reward function we define.

We don’t know the intermediate steps (or if those kinds of _steps_ even exist), which makes it highly unreliable and almost impossible to understand. We also cannot guarantee whether it is aligned with users or not. Even if it’s aligned with humans, for domains that do NOT have obvious metrics (e.g., feelings or emotions) the results might become highly unreliable. Lastly, even if we somehow make the model capable of assessing human feelings and pleasing that person, I’m not sure if that is fundamentally _good_ for us. Since humans are prone to immediate pleasure over long-term gain/well-being, AI’s attempt to please us or manipulate us could negatively affect us.

So while I believe RL has the potential to enable novel developments and breakthroughs within science and technology where metrics can be quantified and observed more objectively, I don’t think RL alone is the way to AGI/ASI (too expensive, too unreliable).

I believe it will do GREAT in code generation (finding new algorithms), chip design, scientific discovery (e.g., biology), gaming, and finding strategies for well-defined A-to-B problems, and I am very excited about it. However, beyond that, value-related applications seem too dangerous.

For the same reason, I think AI agents and robot learning should be based on more explainable systems such as VLA (Vision–Language–Action) rather than RL, since VLA is more scalable, reliable, and understandable. While I am skeptical about current RL, maybe there will be a breakthrough in reliable, aligned RL which I am looking forward to.