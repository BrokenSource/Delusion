---
title: About
icon: material/menu
---

## Standards

In an effort to minimize [xkcd 927](https://xkcd.com/927/), only abstractions that provide clear value or represent shared semantics across providers are made, proxying or extending native packages whenever available.

_For example, the Options class for ollama models shall only apply to itself:_

```python
import ollama

from delusion.chat.option.ollamax import Ollama
from delusion.chat.option.openai import OpenAI

local = Ollama(model="gemma4:e2b")
cloud = OpenAI(model="gpt-whatever")

isinstance(local.options, ollama.Options) # True
isinstance(cloud.options, ollama.Options) # False
```

Although both support `.temperature = 0.0`, the intended use is:

```python
# Individual settings
if os.getenv("PRODUCTION", None):
    chat = OpenAI(model=...)
    chat.options.temperature = 0.0
else:
    chat = Ollama(model=...)
    chat.options.temperature = 0.0

# Shared interface
chat.send(...)
chat.generate(schema=...)
```

Same for models: rather than over-abstracting capabilities, quantization, names, variants, and other provider-specific details, some code duplication is natural to keep it minimal and decoupled.

Conversely, `Message[T]` is abstracted because it represents a common semantic across providers.
