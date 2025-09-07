from interface import GameCommand

class AttackCommand(GameCommand):
    def __init__(self):
        self.mana_cost = 10
        self.damage = 25
        self.exp_gained = 15
        self.old_mana = 0
        self.old_exp = 0
    
    def can_execute(self, actor):
        return actor.mana >= self.mana_cost
    
    def execute(self, actor):
        if not self.can_execute(actor):
            print(f"{actor.name} not enough mana to attack!")
            return
        
        self.old_mana = actor.change_mana(-self.mana_cost)
        self.old_exp = actor.add_experience(self.exp_gained)
        print(f"{actor.name} attacks for {self.damage} damage! (Mana: {actor.mana})")
    
    def undo(self, actor):
        actor.mana = self.old_mana
        actor.experience = self.old_exp
        print(f"{actor.name} undid attack (Mana restored: {actor.mana})")