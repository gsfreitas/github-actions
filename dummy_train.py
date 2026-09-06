import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sys

def main():
    print("Iniciando o Dummy Train")

    try:
        # Carrega o dataset processado
        df = pd.read_csv('dataset_processed.csv')

        # Separa as features e o target
        X = df[['feature1', 'feature2']]
        y = df['target']

        # Treina um modelo rápido
        clf = DecisionTreeClassifier(max_depth=2)
        clf.fit(X, y)

        score = clf.score(X, y)
        print(f"Modelo treinado com sucesso! Acurácia no conjunto de treinamento: {score:.2f}")
        sys.exit
    except Exception as e:
        print(f"ERRO DE COMPUTAÇÃO: o códico do modelo falhou. Detalhes: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
