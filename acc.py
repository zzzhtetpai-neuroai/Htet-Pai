def multiplication(a, b):
    return a * b   

def division(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b    

def addition(a, b):
    return a + b

def subtraction(a, b):
    return a - b    

def power(a, b):
    return a ** b