# Machine-learned models for interaction parameters for multi-fluid models 

## Contributors

Gustavo Chaparro, Imperial College

Ian Bell, formerly at NIST, now Thistle Consultants LLC

(hence the gcib at the end of the package name)

## Note 

These models are under development and subject to change. When reporting your use of these models, make sure you cite the precise version of the library you used, for instance version 0.0.2

## Build sdist and wheels

```
pipx run build
```

## Install

```
pip install .
```
or 
```
pip install git+https://github.com/ibell/gammatml.git
```

## Example

``` python
>>> import gammatmlgcib.average_numpy as avg
>>> db = avg.GammaTModelsDatabase()
>>> db.get_gammaTs(key="TempInd_reg4", InChI_i="InChI=1S/CH2F2/c2-1-3/h1H2", InChI_j="InChI=1S/C2HF5/c3-1(4)2(5,6)7/h1H", T_K=-1)
[0.988520303628153, 0.9548575797898771, 0.974209660030783, 0.9701747034712807, 0.9907890788585142]
```
