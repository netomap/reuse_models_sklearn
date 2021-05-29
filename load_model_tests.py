from time import time
inicio = time()

from ClasseUtilitarios import Utilitarios

utilitario = Utilitarios()
utilitario.import_parameters('ensemble.joblib')

teste = [
    [100, 'admin.', 'married', 'secondary', 'no', 2343, 'yes', 'no',
       'unknown', 5, 'may', 1042, 1, -1, 0, 'unknown']
]

resultado = utilitario.predict_function(teste)

tempo = time() - inicio

print (resultado)
print ('Simulacao realizada em {:0.2f} segundos'.format(tempo))