# Desafio_ML_DIO_Metricas
Link do colab alterado:
https://colab.research.google.com/drive/16CeGhpcAC7VDyTryGpNr5xD6wSiE9v9i#scrollTo=SHkdG_lehhMU

# Resultados
Métricas por classe:
Classe 0:
  Precisão: 0.99
  Sensibilidade: 0.99
  Especificidade: 1.00
  F-Score: 0.99
Classe 1:
  Precisão: 0.99
  Sensibilidade: 1.00
  Especificidade: 1.00
  F-Score: 1.00
Classe 2:
  Precisão: 0.99
  Sensibilidade: 0.98
  Especificidade: 1.00
  F-Score: 0.99
Classe 3:
  Precisão: 1.00
  Sensibilidade: 0.98
  Especificidade: 1.00
  F-Score: 0.99
Classe 4:
  Precisão: 0.99
  Sensibilidade: 0.98
  Especificidade: 1.00
  F-Score: 0.99
Classe 5:
  Precisão: 0.98
  Sensibilidade: 0.99
  Especificidade: 1.00
  F-Score: 0.99
Classe 6:
  Precisão: 0.99
  Sensibilidade: 0.99
  Especificidade: 1.00
  F-Score: 0.99
Classe 7:
  Precisão: 0.98
  Sensibilidade: 0.99
  Especificidade: 1.00
  F-Score: 0.99
Classe 8:
  Precisão: 1.00
  Sensibilidade: 0.99
  Especificidade: 1.00
  F-Score: 0.99
Classe 9:
  Precisão: 0.98
  Sensibilidade: 1.00
  Especificidade: 1.00
  F-Score: 0.99

Acurácia geral: 0.99

# Alterações no colab:
No primeiro comando foi alterada a versão do tensorflow para:
!pip install tensorflow-gpu==2.10.0
No calculo das probabilidades o comando esta obsoleto estão foi alterado para:
y_true=test_labels
y_pred_probabilities = model.predict(test_images)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Dentro do colab foi adicionado o seguinte código para calcular as métricas:
'''código
  def calcular_metricas(con_mat):
      # Soma total de todos os elementos
      total = np.sum(con_mat)
    
      # Soma da diagonal principal Verdadeiro Positivo(VP)
      VP = np.diag(con_mat)
    
      # Falsos positivos (FP) para cada classe
      FP = np.sum(con_mat, axis=0) - VP
    
      # Falsos negativos (FN) para cada classe
      FN = np.sum(con_mat, axis=1) - VP
    
      # Verdadeiros negativos (VN) para cada classe
      VN = total - (VP + FP + FN)
    
      # Calcular métricas
      acuracia = np.sum(VP) / total
      precisao = VP / (VP + FP)
      sencibilidade = VP / (VP + FN)
      especificidade = VN / (VN + FP)
      f_score = 2 * (precisao * sencibilidade) / (precisao + sencibilidade)
    
    
      return acuracia, precisao, sencibilidade, especificidade, f_score

  # Usar a matriz de confusão normalizada
  acuracia, precisao, sencibilidade, especificidade, f_score = calcular_metricas(con_mat)

  # Exibir resultados
  classes = ['Classe 0', 'Classe 1', 'Classe 2', 'Classe 3', 'Classe 4', 'Classe 5', 'Classe 6', 'Classe 7', 'Classe 8', 'Classe 9']  # Ajustar de acordo com as classes reais
  print("Métricas por classe:")
  for i, cls in enumerate(classes):
      print(f"{cls}:")
      print(f"  Precisão: {precisao[i]:.2f}")
      print(f"  Sensibilidade: {sencibilidade[i]:.2f}")
      print(f"  Especificidade: {especificidade[i]:.2f}")
      print(f"  F-Score: {f_score[i]:.2f}")

  print(f"\nAcurácia geral: {acuracia:.2f}")
'''
