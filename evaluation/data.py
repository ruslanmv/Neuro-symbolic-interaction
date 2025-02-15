evaluation_data = [
    # Custom statements
    ("Does battery_1 cause failure in electric_engine_1?", "battery_1.CausesFailure(electric_engine_1)", True),
    # True statements with variations
    ("What causes piston failure in oil engines?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),
    ("How does a piston lead to oil engine failure?", "The piston leads to the oil engine failing.", "piston_1.CausesFailure(oil_engine_1)", True),
    ("Why does a piston cause oil engine failure?", "Failure of the oil engine is caused by the piston.", "piston_1.CausesFailure(oil_engine_1)", True),
    ("In what specific ways can a piston cause failure in an oil engine?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),
    ("What are the signs of a failing piston in an oil engine?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),
    ("How can the condition of a piston in an oil engine be monitored and assessed?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),
    ("What factors contribute to the increased risk of piston failure in oil engines?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),
    ("How does piston design and material choice impact the likelihood of failure in oil engines?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),
    ("What maintenance practices can help prolong the life of pistons in oil engines?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),

    ("What causes oil pump failure in oil engines?", "Oil pump causes failure of oil engine", "oil_pump_1.CausesFailure(oil_engine_1)", True),
    ("How does an oil pump lead to oil engine failure?", "The oil pump leads to the oil engine failing.", "oil_pump_1.CausesFailure(oil_engine_1)", True),
    ("Why does an oil pump cause oil engine failure?", "Failure of the oil engine is caused by the oil pump.", "oil_pump_1.CausesFailure(oil_engine_1)", True),

    ("What causes battery failure in electric engines?", "Battery causes failure of electric engine", "battery_1.CausesFailure(electric_engine_1)", True),
    ("How does a battery lead to electric engine failure?", "The battery leads to the electric engine failing.", "battery_1.CausesFailure(electric_engine_1)", True),
    ("Why does a battery cause electric engine failure?", "Failure of the electric engine is caused by the battery.", "battery_1.CausesFailure(electric_engine_1)", True),

    ("What causes motor failure in electric engines?", "Motor causes failure of electric engine", "motor_1.CausesFailure(electric_engine_1)", True),
    ("How does a motor lead to electric engine failure?", "The motor leads to the electric engine failing.", "motor_1.CausesFailure(electric_engine_1)", True),
    ("Why does a motor cause electric engine failure?", "Failure of the electric engine is caused by the motor.", "motor_1.CausesFailure(electric_engine_1)", True),

    # Additional true statements (repeats for more examples)
    ("What causes piston failure in oil engines?", "Piston causes failure of oil engine", "piston_1.CausesFailure(oil_engine_1)", True),
    ("What causes oil pump failure in oil engines?", "Oil pump causes failure of oil engine", "oil_pump_1.CausesFailure(oil_engine_1)", True),
    ("What causes battery failure in electric engines?", "Battery causes failure of electric engine", "battery_1.CausesFailure(electric_engine_1)", True),
    ("What causes motor failure in electric engines?", "Motor causes failure of electric engine", "motor_1.CausesFailure(electric_engine_1)", True),

    # False statements
    ("Why would a battery cause an oil engine to fail?", "Battery causes failure of oil engine", "battery_1.CausesFailure(oil_engine_1)", False),
    ("Why would an oil pump cause an electric engine to fail?", "Oil pump causes failure of electric engine", "oil_pump_1.CausesFailure(electric_engine_1)", False),
    ("Why would a motor cause an oil engine to fail?", "Motor causes failure of oil engine", "motor_1.CausesFailure(oil_engine_1)", False),
    ("Why would a piston cause an electric engine to fail?", "Piston causes failure of electric engine", "piston_1.CausesFailure(electric_engine_1)", False),

    ("Why would a battery cause an oil engine to fail?", "Battery causes failure of oil engine", "battery_1.CausesFailure(oil_engine_1)", False),
    ("Why would a motor cause an oil engine to fail?", "Motor causes failure of oil engine", "motor_1.CausesFailure(oil_engine_1)", False),
    ("Why would a piston cause an electric engine to fail?", "Piston causes failure of electric engine", "piston_1.CausesFailure(electric_engine_1)", False),
    ("Why would a battery cause an oil engine to fail?", "Battery causes failure of oil engine", "battery_1.CausesFailure(oil_engine_1)", False),
    ("Why would an oil pump cause an electric engine to fail?", "Oil pump causes failure of electric engine", "oil_pump_1.CausesFailure(electric_engine_1)", False),

    ("Why would a battery cause an oil engine to fail?", "Battery causes failure of oil engine", "battery_1.CausesFailure(oil_engine_1)", False),
    ("Why would a motor cause an oil engine to fail?", "Motor causes failure of oil engine", "motor_1.CausesFailure(oil_engine_1)", False),
    ("Why would a piston cause an electric engine to fail?", "Piston causes failure of electric engine", "piston_1.CausesFailure(electric_engine_1)", False),
    ("Why would a battery cause an oil engine to fail?", "Battery causes failure of oil engine", "battery_1.CausesFailure(oil_engine_1)", False),
    ("Why would an oil pump cause an electric engine to fail?", "Oil pump causes failure of electric engine", "oil_pump_1.CausesFailure(electric_engine_1)", False),

    ("Why would a battery cause an oil engine to fail?", "Battery causes failure of oil engine", "battery_1.CausesFailure(oil_engine_1)", False),
    ("Why would a motor cause an oil engine to fail?", "Motor causes failure of oil engine", "motor_1.CausesFailure(oil_engine_1)", False),
    ("Why would a piston cause an electric engine to fail?", "Piston causes failure of electric engine", "piston_1.CausesFailure(electric_engine_1)", False),
    ("Why would a battery cause an oil engine to fail?", "Battery causes failure of oil engine", "battery_1.CausesFailure(oil_engine_1)", False),
    ("Why would an oil pump cause an electric engine to fail?", "Oil pump causes failure of electric engine", "oil_pump_1.CausesFailure(electric_engine_1)", False),
]
evaluation_data_test = [
    # Custom statements
    ("Does battery_1 cause failure in electric_engine_1?", "battery_1.CausesFailure(electric_engine_1)", True),
    ("Why would an oil pump cause an electric engine to fail?", "Oil pump causes failure of electric engine", "oil_pump_1.CausesFailure(electric_engine_1)", False),

 ]
