#Problemparameter:
Objects = [(Weight, Value), ...] #Gegenstandsliste, nach Gewicht sortiert
Maximalgewicht = Ausgangswert #Maximalgewicht, das haltbar ist

#Aufstellung des Lösungsspeichers := 'M' als n = 2 -D 'Array'
M = [[0 for i in range(Maximalgewicht)] for j in range(len(Objects))]
#Eingabe: Ein Zustand: s := (k, w)
#k: Betrachteter Gegenstand, w: Verbleibendes Gesamtgewicht
#Effekt: Der Lösunsspeicher wird entsprechend der Wertefunktion verändert
#Ergebnis: (Veränderter) Wert wird ausgegeben
def v(k, w):
    if k == -1: #Base-Case: kein Objekt passt mehr
        M[k][w] = 0   #Base-Case-Return: keine Belohnung
    elif Objects[k][0] > w:     #Gezwungene Handlung: Objekt passt nicht
        M[k][w] = v((k - 1, w)) #Nächstkleineres Objekt wird betrachtet, Gewicht bleibt
    else: #Es gibt die freie Wahl, ob Objekt 'eingesteckt' wird
        M[k][w] = max(Objects[k][1] + v(k-1, w-Objects[k][0]), #Erfahrene Belohnung addiert
                   v(k - 1, w)) #Keine Belohnung, Gewicht beibehalten
    return M[k][w]
Lösungswert = v(len(M) - 1, Maximalgewicht)
piOpt = []
#Effekt: Geht die Entscheidungsfolge der Optimallösung anhand Lösungsspeicher nach
#(Effektiv: Simuliert Entscheidungen, wählt die mit besten Ausgang anhand v)
def constructPiOpt(k, w):
    if k == -1:
        return
    elif Objects[k][0] > w:
        constructPiOpt(k - 1, w)
    elif Objects[k][1] + M[k - 1, w-Objects[k][0]] > M[k-1][w]:
        piOpt.append(Objects[k])
    else:
        constructPiOpt(k-1, w)

