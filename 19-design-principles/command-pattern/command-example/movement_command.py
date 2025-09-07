from interface import GameCommand

class MoveCommand(GameCommand):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy
        self.old_position = None
    
    def execute(self, actor):
        self.old_position = actor.move_to(actor.x + self.dx, actor.y + self.dy)
        direction = self._get_direction()
        print(f"🚶 {actor.name} moved {direction} to ({actor.x}, {actor.y})")
    
    def undo(self, actor):
        if self.old_position:
            actor.move_to(*self.old_position)
            print(f"↩️  {actor.name} moved back to ({actor.x}, {actor.y})")
    
    def _get_direction(self):
        if self.dx == 1: return "right"
        if self.dx == -1: return "left"
        if self.dy == 1: return "up"
        if self.dy == -1: return "down"
        return "nowhere"

class MoveUpCommand(MoveCommand):
    def __init__(self): super().__init__(0, 1)

class MoveDownCommand(MoveCommand):
    def __init__(self): super().__init__(0, -1)

class MoveLeftCommand(MoveCommand):
    def __init__(self): super().__init__(-1, 0)

class MoveRightCommand(MoveCommand):
    def __init__(self): super().__init__(1, 0)