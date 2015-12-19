# A simple motion tracker using OpenCV

## Abandoned WIP!

Code is in `tests.cpp`, the class and header files were just the start of a cleaner implementation.

On Debian Squeeze or Wheezy (can't remember which and didn't test on any other distribution), if you get an `undefined references` error, use that to compile:

```shell
g++ tests.cpp -o tests `pkg-config opencv --cflags --libs`
```
