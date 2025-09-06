### Builder pattern

Builder is a creational design pattern that lets you construct complex objects step by step. The pattern allows you to produce different types and representations of an object using the same construction code.

Instead of calling a big constructor with many parameters, the builder provides methods to configure each part, and then a final .build() method to assemble the object.

#### When to use

1. When creating objects with lots of optional parameters.
2. When object creation requires complex steps.
3. To make the code more readable compared to long constructors.