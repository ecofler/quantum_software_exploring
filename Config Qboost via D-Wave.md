# Qboost - Quantum version of boosting algorithm

To use Qboost as a Python script in its standard formulation, clone the 
repository https://github.com/dwave-examples/qboost.git.

Inside of it, there are the qboost.py script that defines the algorithm itself, the demo.py 
that can be used to have a run example and the rest of the scripts that are dependencies 
of qboost.py. Everything else is described in the README.md of the qboost repository,
including the required package dependencies in requirements.txt.

To use qboost.py, some preparation steps are necessary: 
 - install the requirements specified in the 
aforementioned requirements.txt
    - ```pip install matplotlib ``` 
    - ```pip install tabulate```
    - ```pip install scikit-learn```
    - ```pip install dwave-ocean-sdk --user```
 - configure the D-Wave Solver API in the following way
    - run ```
          dwave config create
           ``` 
    and when prompt for inputs, leave the default ``` Profile ``` name and insert the 
      ``` Authentication token ``` code that can be found in the AuthCodeAlmaviva.txt 
      file provided with this guide.
      
To test the success of the previous passages, you can run `python demo.py digits` which
will test the qboost.py algorithm on the well-known handwritten digits data set; if 
everything is working, you should see a standard output similar to 

>Number of features: 64 
> 
>Number of training samples: 216
> 
> Number of test samples: 144
> 
> Number of selected features: 23
> 
> Score on test set: 0.993

If errors related to D-Wave API arise, make sure that the access to D-Wave machines is 
correctly configured by running `dwave sample --random-problem`, you should see an output
similar to
> Using endpoint: https://my.dwavesys.url/
> 
> Using solver: My_DWAVE_2000Q
> 
> Using qubit biases: {0: -1.0345257941434953, 1: -0.5795618633919246, 2: 0.9721956399428491, 3: 1....
> 
> Using qubit couplings: {(1634, 1638): 0.721736584181423, (587, 590): 0.9611623181258304, (642, 64...
> 
> Number of samples: 1
> 
> Samples: [[1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1,...
> 
> Occurrences: [1]
> 
> Energies: [-2882.197791239335]

Once the installing is done, we can use Qboost.py. In this benchmarking version, the booster works on
a D-Wave solver which takes in input at most 20000 weak classifiers, and uses Decision Stump classifiers 
implemented with Scikit-learn. Also, a regularization parameter `lambda` is fixed, but it can be improved 
with a cross validation which does a parameter sweep and chooses the best one, as described in the qboost
repository README.

To run the script, import the function `from qboost import QBoostClassifier` and use it in the following 
way` QBoostClassifier(X_train, y_train, lam)`, where `X_train` and `y_train` are the testing set features 
and labels to provide and `lam` the regularization parameter, which can be set to 0.01 as a default 
value.


