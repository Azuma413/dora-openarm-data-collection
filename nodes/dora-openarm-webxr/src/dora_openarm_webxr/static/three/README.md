# Vendored three.js (openarm-quest-teleop addition)

The 3D view (`view: scene` / `view: scene+panel`, PATCHES.md 15) draws the
simulated cell with three.js. The library is vendored here rather than fetched
from a CDN because the headset and the node share a LAN that need not have a
route to the Internet, and because the page must be byte-for-byte the one the
node serves.

| File | Origin in the npm package `three@0.179.1` | Change |
| --- | --- | --- |
| `three.module.min.js` | `build/three.module.min.js` | none (byte-identical) |
| `three.core.min.js` | `build/three.core.min.js` | none (byte-identical) |
| `GLTFLoader.js` | `examples/jsm/loaders/GLTFLoader.js` | two import specifiers rewritten to relative paths (`'three'` → `'./three.module.min.js'`, `'../utils/BufferGeometryUtils.js'` → `'./BufferGeometryUtils.js'`) so the page needs no import map |
| `BufferGeometryUtils.js` | `examples/jsm/utils/BufferGeometryUtils.js` | one import specifier rewritten the same way |
| `LICENSE` | `LICENSE` | none |

- Version: **0.179.1** (`REVISION` 179), npm integrity
  `sha512-5y/elSIQbrvKOISxpwXCR4sQqHtGiOI+MKLc3SsBdDXA2hz3Mdp3X59aUp8DyybMa34aeBwbFTpdoLJaUDEWSw==`
- Licence: **MIT**, Copyright © 2010-2025 three.js authors — full text in
  `LICENSE` next to this file.
- Upstream: <https://github.com/mrdoob/three.js> (`r179`).

SHA-256 of the files as vendored:

```
06552c54e4071fbc7305117aafe6765d92c5d2a2a83507d4f05b9bf4f3d4d463  three.module.min.js
79f2b4f58d3e99a9948a4d3b7f6d5c2daf705bdefe9fb82ebec715623966551c  three.core.min.js
6b0d1e699dd2762a069c4eb1f60cbb843a6ff05fc5fb5a055f5509326a3efd48  GLTFLoader.js
fcdbefee37e0ca2c26b7c1da5d9362560983871458198c6d3e1ccf56f84aadc4  BufferGeometryUtils.js
bfe119ea4fd413f5f7ca3fcd63adb0c4a073ed39daa2fe7d3e6b769e21272601  LICENSE
```

To update: `npm pack three@<version>`, copy the four files, redo the import
rewrites, refresh the table, the hashes and the NOTICE entry, and re-run
`tests/web/scene3d.test.mjs`.

Nothing else on the page imports three.js: `ar.js` loads `scene3d.js` with a
dynamic `import()` only when the node asks for a 3D view, so the camera-panel
views (`mono`, `stereo`, `none`) download none of this.
