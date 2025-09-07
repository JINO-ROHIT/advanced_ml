### Command pattern (known as action transaction)

instead of calling a method directly, you wrap the request inside a command object, which has a standard interface (like execute()). Then, the caller just triggers the command, without knowing what the actual action is.

instead of shouting 'turn on the light' directly at a light bulb, you write 'turn on the light' on a piece of paper (the command object) and hand it to someone who can execute it when appropriate.

Key Components
1. Command Interface - Defines what all commands must be able to do (usually execute() and undo())
2. Concrete Commands - Specific actions like "Move Left", "Attack", "Save File" that implement the interface
3. Receiver - The object that actually performs the work (the game player, text editor, etc.)
4. Invoker - The object that holds and executes commands (remote control, input handler, menu system)
