cursor = (('person',123,23434),('person',343,65656), ('car', 3434, 4544),('car',454,4545))
_type =""
rows=[]
print(cursor)    
for record in cursor:
        type = record[0]
        if(type != _type): 
            rows.append({'label':record[0],'values': 
            [ {'x':v[1], 'y':v[2]} for v in list(filter( lambda x : x[0] == type , cursor))] })
        _type=type
print(rows)

