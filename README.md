# spherical-MDS

Graph Layouts on the sphere

```
usage: layout.py [-h] [--max_iter MAX_ITER]
                [--learning_rate LEARNING_RATE] [--output OUTPUT]
                input_graph

Read a graph, and produce a layout with tsNET(*).

positional arguments:
  input_graph

optional arguments:
  -h, --help            show this help message and exit

  --opt_scale           Whether to optimizing the scale of the data.
                        Defaults to 0, which will use a heuristic.
                        A 1 will use the optimization.

  --epsilon -e          Threshold for convergence.

  --learning_rate LEARNING_RATE, -l LEARNING_RATE
                        If given a number, will use a fixed schedule. Will also accept
                        'fraction' 'sqrt' and 'convergent' schedules. Defaults to 'convergent'

  --max_iter            Maximum number of iterations.

  --output OUTPUT, -o OUTPUT
                        Save layout to the specified file.
```

Example:
```bash
# Read the input graph dwt_72, and save the output in ./out.json
python3 layout.py graphs/cube.txt --output out.json
```

# Dependencies

* `python3`
* [`numpy`](http://www.numpy.org/)
* [`matplotlib`](https://matplotlib.org/)
* [`graph-tool`](https://graph-tool.skewed.de/)
* [`numba`](http://deeplearning.net/software/theano/)
* [`scikit-learn`](http://scikit-learn.org/stable/)

# Example
