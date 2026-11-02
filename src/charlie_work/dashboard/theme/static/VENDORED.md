# Vendored static assets

| asset | version | source | integrity |
|---|---|---|---|
| `htmx.min.js` | htmx.org 2.0.11 | https://unpkg.com/htmx.org@2.0.11/dist/htmx.min.js | sha384-2OatzQy1H+Zd/IIrjr1TcuDGqLXeHhbooAyJY1KdQMKnr4LZ22k31GBLdYKHmVjg |
| `fonts/fraunces-latin-wght-normal.woff2` | @fontsource-variable/fraunces 5.3.0 | swole `packages/living-journal/node_modules` | OFL-1.1, `fonts/OFL-Fraunces.txt` |
| `fonts/fraunces-latin-wght-italic.woff2` | @fontsource-variable/fraunces 5.3.0 | same | OFL-1.1 |
| `fonts/inter-latin-wght-normal.woff2` | @fontsource-variable/inter 5.3.0 | swole `packages/living-journal/node_modules` | OFL-1.1, `fonts/OFL-Inter.txt` |

## htmx verification

The sha384 above is the `integrity` value published on the official installation page
(https://htmx.org/docs/#installing, jsDelivr snippet for `htmx.org@2.0.11/dist/htmx.min.js`),
fetched 2026-10-01, and equals the digest computed locally over the vendored bytes
(`openssl dgst -sha384 -binary htmx.min.js | openssl base64 -A`).

2.0.11 rather than the originally planned 2.0.4: the official page publishes a hash only for
2.0.11, so 2.0.4 could not be verified from an official source. Re-verify against the docs
page on every bump.

The dashboard makes no outbound requests: fonts and htmx are served from `/static/`.
