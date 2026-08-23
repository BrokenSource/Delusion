---
title: Changelog
icon: material/file-document-edit
tags:
- About
- Changelog
- Versions
---

<style>ul li {line-height: 1.1}</style>

### :package: v0.4.1 <small>Unreleased</small> {#v0.4.1}

!!! quote ""
    - Rework audio.cpp architecture and refactor subpackage
    - Default OpenMP workers to `os.cpu_count() or 4`
    - Support and abstract Stable Audio 3 model

### :package: v0.4.0 <small>August 16, 2026</small> {#v0.4.0}

!!! quote ""
    - Add initial [audio.cpp](https://github.com/0xShug0/audio.cpp) wrapper classes
    - Add `py.typed` file marker for PEP 561 compliance in linters

### :package: v0.3.1 <small>August 4, 2026</small> {#v0.3.1}

!!! quote ""
    - Add initial project logo
    - Swap [`smartdirs`](https://github.com/BrokenSource/SmartDirs) for `platformdirs` until my package is more stable
    - Refactor and make ollama cache methods opt-in
