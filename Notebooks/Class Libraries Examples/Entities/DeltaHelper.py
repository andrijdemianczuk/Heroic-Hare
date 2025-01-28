from abc import abstractmethod, abstractproperty, ABC

class DeltaHelper(object):
    
    def __init__(self, name:str):
        self.name = name

    def getMessage(self) -> str:
        return (f"Hi {self.name}. Welcome to the world. Go make something cool!")
    
class SubHelper(DeltaHelper):

    def __init__(self, name:str):
        super().__init__(name=name)

    def sayGoodbye(self):
        return(f"Goodbye {self.name}.")
    

class DeltaBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def getSession(self):
        return "foo"

class DeltaConcrete(DeltaBase):

    def __init__(self):
        pass

    ### Required since we're creating a concrete class based off the abstract base class
    def getSession(self):
        return super().getSession()

    ### Functionality extended to add behavior
    def describeSession(self):
        return "blah"