# Sources and licensing

Source code and documentation authored for this project are provided under
the [MIT license](../LICENSE).

## References

This project combines a ContactSDF-inspired geometry representation with an
explicit complementarity-free contact response:

- Wanxin Jin, [Complementarity-Free Multi-Contact Modeling and Optimization for Dexterous Manipulation](https://arxiv.org/abs/2408.07855).
  [Code](https://github.com/asu-iris/Complementarity-Free-Dexterous-Manipulation).
- Wen Yang and Wanxin Jin, [ContactSDF: Signed Distance Functions as Multi-Contact Models for Dexterous Manipulation](https://arxiv.org/abs/2408.09612).
  [Code](https://github.com/asu-iris/ContactSDF).

## Dependencies and demo

NumPy, CasADi/IPOPT, MuJoCo, Matplotlib, Pillow, and pytest retain their own
licenses and notices.

The scene uses MuJoCo primitives. The demo is generated with this project's
`run.py` and `render.py` scripts.
