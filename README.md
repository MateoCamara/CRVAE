# Complex Recurrent VAE

A TensorFlow implementation of a **complex-valued recurrent variational autoencoder**
for speech enhancement, reproducing results from
[Complex Recurrent Variational Autoencoder for Speech Enhancement](https://arxiv.org/pdf/2204.02195.pdf)
(Xie, Arildsen & Tan, 2022).

The model encodes noisy complex STFT spectra into a complex Gaussian latent space
(sampled with a complex reparameterisation trick and regularised by a closed-form
complex KL divergence) and decodes an enhanced complex spectrum, which is then
reconstructed into a waveform.

## Repository layout

| Path | Contents |
| ---- | -------- |
| `complexnn/` | Complex-valued neural-network layers (dense, GRU, batch/layer norm, initializers, FFT, pooling). Third-party — see [Acknowledgments](#acknowledgments). |
| `libs/costs.py` | Complex VAE cost functions: complex Gaussian KL divergence, complex log-likelihood and the complex / real reparameterised samplers. |
| `runners/` | Shared TensorFlow session configuration. |
| `tools/audio.py` | STFT / inverse-STFT helpers and complex-spectrum-to-waveform reconstruction. |
| `scripts/` | The two training / evaluation entry points (see below). |

### Entry points (`scripts/`)

- **`CLAAUDIA_speech_enhancement_CVAE1.py`** — the complex *recurrent* VAE: the
  encoder and decoder are Stiefel-constrained complex GRUs (`StiefelGatedRecurrentUnit`).
- **`CLAAUDIA_speech_enhancement_complex_nonlinear_VAE.py`** — a non-recurrent
  baseline: complex fully-connected layers (`ComplexDense`) with a `mod_relu`
  non-linearity.

## Dependencies

This is legacy research code and needs an old stack:

- **Python 3.7**
- [**TensorFlow 1.15**](https://www.tensorflow.org/) (TF1 graph API, used through `tensorflow.compat.v1`)
- [SciPy](https://scipy.org/) and [NumPy](https://numpy.org/), plus `six`

```bash
pip install -r requirements.txt   # in a Python 3.7 environment
```

See `requirements.txt` for the version bounds and why they matter.

## Data

The speech datasets are **not** included in this repository. Each script reads
MATLAB `.mat` files whose `feats` variable holds complex STFT spectra (the first
column is dropped, keeping `N_dim = 200` frequency bins). Data is organised by noise
type (`bbl`, `caf`, `ssn`, `str`) with `train` / `dev` / `test` splits:

```
<noisy_speech_path>/<noise_type>/{train,dev,test0,test3,test-3,test-6,test6}.mat
<clean_speech_path>/{train,dev,test}.mat
```

Before running, set the (currently empty) path variables at the top of each script
— `noisy_speech_path`, `clean_speech_path`, `model_dir`, `exp_dir`, `model_path`,
`output_path` — to point at your own locations.

## Usage

```bash
cd scripts
python CLAAUDIA_speech_enhancement_CVAE1.py
```

Each script builds a TF1 graph, trains with early stopping on the development set,
saves the best checkpoint, and then enhances the test set — writing both the
enhanced complex spectra (`.mat`) and the reconstructed audio (`.wav`).

## Acknowledgments

- The `complexnn/` package is built on the **Deep Complex Networks** code by
  **Chiheb Trabelsi, Olexa Bilaniuk et al.** (MIT-licensed); the original author
  headers are preserved in each file. It additionally includes a Stiefel-constrained
  complex GRU.
- The model and training recipe reproduce **Xie, Arildsen & Tan (2022)** (cited below).

## Citation

To cite the original work, please use:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2204.02195,
  doi = {10.48550/ARXIV.2204.02195},
  url = {https://arxiv.org/abs/2204.02195},
  author = {Xie, Yuying and Arildsen, Thomas and Tan, Zheng-Hua},
  title = {Complex Recurrent Variational Autoencoder for Speech Enhancement},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## License

This repository does not declare an explicit license. The vendored `complexnn/` code
retains its original (MIT) terms and author attribution; the rest of the code
reproduces the method cited above. If you wish to reuse it, please contact the author
and credit the original works listed under [Acknowledgments](#acknowledgments).
