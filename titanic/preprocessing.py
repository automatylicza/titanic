import pandas as pd

def load_data(base_dir):
    """Funkcja wczytująca dane."""
    train_data = pd.read_csv(base_dir / 'data/external/train.csv')
    test_data = pd.read_csv(base_dir / 'data/external/test.csv')
    return train_data, test_data

def preprocess_data(train_data, test_data):
    """Funkcja przetwarzająca dane."""
    # Uzupełnianie braków
    train_data['Age'] = train_data.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    test_data['Age'] = test_data.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
    test_data['Embarked'] = test_data['Embarked'].fillna(test_data['Embarked'].mode()[0])
    test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

    # Tworzenie nowych cech
    train_data['HasCabin'] = train_data['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
    test_data['HasCabin'] = test_data['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
    
    age_bins = [0, 10, 19, 35, 60, 120]
    age_labels = ['Child', 'Teenager', 'Adult', 'Middle-aged', 'Senior']
    train_data['AgeGroup'] = pd.cut(train_data['Age'], bins=age_bins, labels=age_labels)
    test_data['AgeGroup'] = pd.cut(test_data['Age'], bins=age_bins, labels=age_labels)
    
    fare_bins = [-1, 0, 20, 50, 100, 300, 600]
    fare_labels = ['Free', 'Very Low', 'Low', 'Medium', 'High', 'Very High']
    train_data['FareGroup'] = pd.cut(train_data['Fare'], bins=fare_bins, labels=fare_labels)
    test_data['FareGroup'] = pd.cut(test_data['Fare'], bins=fare_bins, labels=fare_labels)
    
    # Kodowanie zmiennych
    train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
    test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

    train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)
    test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)

    age_group_map = {'Child': 0, 'Teenager': 1, 'Adult': 2, 'Middle-aged': 3, 'Senior': 4}
    train_data['AgeGroup'] = train_data['AgeGroup'].map(age_group_map)
    test_data['AgeGroup'] = test_data['AgeGroup'].map(age_group_map)

    fare_group_map = {'Free': 0, 'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5}
    train_data['FareGroup'] = train_data['FareGroup'].map(fare_group_map)
    test_data['FareGroup'] = test_data['FareGroup'].map(fare_group_map)

    return train_data, test_data
