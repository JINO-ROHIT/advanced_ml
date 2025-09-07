from abc import ABC, abstractmethod


class GameCommand(ABC):
    @abstractmethod
    def execute(self, actor):
        pass
    
    @abstractmethod
    def undo(self, actor):
        pass
    
    def can_execute(self, actor):
        """Check if command can be executed (e.g., enough mana, valid position)"""
        return True
    
    def get_name(self):
        return self.__class__.__name__.replace('Command', '')