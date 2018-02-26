#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    
    ### your code goes here
    for i in range(0, len(ages)):
        error = abs(predictions[i][0] - net_worths[i][0])
        tup = (ages[i][0], net_worths[i][0], error)
        cleaned_data.append(tup)
    
    cleaned_data = sorted(cleaned_data, key=lambda tup: tup[2])
    
    limit = int(len(cleaned_data)*0.1)
    for j in range(0, limit):
#         print j
        cleaned_data.pop()
        
    print len(cleaned_data)
    
    return cleaned_data

