from abc import ABC, abstractmethod

class Command(ABC):

    @abstractmethod
    def execute(self):
        """[Ceci à valeur d'interface] Méthode qui doit être implémentée par les sous-classes.
        Si elle n'est pas implémentée, une exception sera levée."""
        raise NotImplementedError("La méthode 'execute()' doit être implémentée dans la sous-classe.")