# S3AIPolytope

This is a project that helps to work with the polytopes that correspond to the neural network's linear regions at the input space.
By using code from the project, one can:
- calculate polytope that contains a pre-defined point from the input space;
- check if the calculated region is bounded or not;
- plot polytopes for 2D case.

To start the project:
1) Create folder **data/** :
```shell
$ mkdir data
```
2) Install dependencies:
```shell
$ pip install -r requirements.txt
```
3) Run main.py file to run the project: 
```shell 
$ python main.py
```

The **main.py** file contains following methods:
```python
# method that checks if polytope is bounded
def is_bouded(A):
    ...
```
```python
# method that calculate a polytope for given neural network and point
def get_poly(model, point):
    ...
```
```python
# for a given list of points check if the corresponding regions are bounded or not
def regions_check(model_structure, device, total_points=10):
    ...
```