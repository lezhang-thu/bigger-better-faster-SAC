import numpy as np

rr2_results = dict()
rr2_results['ChopperCommand'] = np.asarray([
    1709.00, 2098.00, 2186.00, 3278.00, 6502.00, 4296.00, 7049.00, 9173.00,
    3420.00, 1390.00
])
rr2_results['Hero'] = np.asarray([
    7502.50, 7529.55, 7299.00, 10300.00, 7868.50, 8184.50, 7900.00, 7856.60,
    6271.20, 7682.80
])

print(rr2_results['ChopperCommand'].mean())
print(rr2_results['Hero'].mean())
