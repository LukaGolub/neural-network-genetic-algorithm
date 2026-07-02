import sys
import math
from collections import Counter, defaultdict
import argparse
import numpy as np

def read_file(filename): #Citanje datoteka
    with open(filename, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            clean = line.strip()
            if clean and not clean.startswith("#"):
                lines.append(clean)

    table = []
    for line in lines:
        table.append(line.split(","))

    header = table[0]
    data = np.array([[float(val) for val in row] for row in table[1:]])
    return header, data


class Mreza: #Klasa koja predstavlja samu mrezu
    def __init__(self, br_ulaza, arh):
        self.arh = arh
        self.tezine = [] #Matrice tezina
        self.pomaci = [] #Vektori pomaka
        #Init tezina i pomaka ovisno o arhitekturi
        if arh == "5s":
            w1 = np.random.normal(0, 0.01, (br_ulaza, 5))
            p1 = np.random.normal(0, 0.01, (1, 5))
            w2 = np.random.normal(0, 0.01, (5, 1))
            p2 = np.random.normal(0, 0.01, (1, 1))
            self.tezine.append(w1)
            self.pomaci.append(p1)
            self.tezine.append(w2)
            self.pomaci.append(p2)

        elif arh == "20s":
            w1 = np.random.normal(0, 0.01, (br_ulaza, 20))
            p1 = np.random.normal(0, 0.01, (1, 20))
            w2 = np.random.normal(0, 0.01, (20, 1))
            p2 = np.random.normal(0, 0.01, (1, 1))
            self.tezine.append(w1)
            self.pomaci.append(p1)
            self.tezine.append(w2)
            self.pomaci.append(p2)

        elif arh == "5s5s":
            w1 = np.random.normal(0, 0.01, (br_ulaza, 5))
            p1 = np.random.normal(0, 0.01, (1, 5))
            w2 = np.random.normal(0, 0.01, (5, 5))
            p2 = np.random.normal(0, 0.01, (1, 5))
            w3 = np.random.normal(0, 0.01, (5, 1))
            p3 = np.random.normal(0, 0.01, (1, 1))
            self.tezine.append(w1)
            self.pomaci.append(p1)
            self.tezine.append(w2)
            self.pomaci.append(p2)
            self.tezine.append(w3)
            self.pomaci.append(p3)

    def sigmoid(self, x): #Altivacijska funkcija
        return 1 / (1 + np.exp(-x))
 
    def prolaz(self, podaci): #Unaprijedni prolaz
        trenutni = podaci
        slojevi = len(self.tezine)
        
        for i in range(slojevi - 1): #Prolaz kroz skrivene slojeve
            W = self.tezine[i]
            P = self.pomaci[i]
            kombinacija = trenutni @ W + P
            trenutni = self.sigmoid(kombinacija)
            
        zadnjeW = self.tezine[-1] #Izlazni sloj
        zadnjeP = self.pomaci[-1]
        izlaz = trenutni @ zadnjeW + zadnjeP
        
        return izlaz

    def get_ravno(self):
        ravno = []
        for i in range(len(self.tezine)):
            tw = self.tezine[i]
            tp = self.pomaci[i]
            
            flatw = tw.flatten()
            for j in range(len(flatw)):
                ravno.append(flatw[j])
                
            flatp = tp.flatten()
            for j in range(len(flatp)):
                ravno.append(flatp[j])
                
        return np.array(ravno)

    def set_ravno(self, niz):
        brojac = 0
        for i in range(len(self.tezine)):
            oblikw = self.tezine[i].shape
            velicinaw = self.tezine[i].size  
            
            komadw = niz[brojac : brojac + velicinaw]
            self.tezine[i] = komadw.reshape(oblikw)
            brojac = brojac + velicinaw
            
            oblikp = self.pomaci[i].shape
            velicinap = self.pomaci[i].size  
            
            komadp = niz[brojac : brojac + velicinap]
            self.pomaci[i] = komadp.reshape(oblikp)
            brojac = brojac + velicinap

#Srednja kvadratna pogreska
def mse(stvarno, predvideno):
    return np.mean((stvarno - predvideno) ** 2)

#Odabir jedinke
def selekcija(populacija, vjerojatnosti):
    r = np.random.rand()
    suma = 0.0
    for i in range(len(populacija)):
        suma += vjerojatnosti[i]
        if r <= suma:
            return populacija[i]
    return populacija[-1]

#Krizanje roditelja
def krizanje(roditelj1, roditelj2):
    return (roditelj1 + roditelj2) / 2.0

#Mutiranje gena dodavanjem gausovog suma
def mutacija(kromosom, pm, skala):
    mutirani = kromosom.copy()
    maska = np.random.rand(len(mutirani)) < pm
    buka = np.random.normal(0, skala, len(mutirani))
    mutirani[maska] += buka[maska]
    return mutirani


def main():
    pars = argparse.ArgumentParser()
    pars.add_argument('--train', type=str, required=True)
    pars.add_argument('--test', type=str, required=True)
    pars.add_argument('--nn', type=str, required=True)
    pars.add_argument('--popsize', type=int, required=True)
    pars.add_argument('--elitism', type=int, required=True)
    pars.add_argument('--p', type=float, required=True)
    pars.add_argument('--K', type=float, required=True)
    pars.add_argument('--iter', type=int, required=True)

    args = pars.parse_args()

    _, train_data = read_file(args.train)
    _, test_data = read_file(args.test)
    #ucitavanje podataka
    X_train, y_train = train_data[:, :-1], train_data[:, -1:]
    X_test, y_test = test_data[:, :-1], test_data[:, -1:]

    ulazi = X_train.shape[1] 

    mreza = Mreza(ulazi, args.nn)
    br_paramatara = mreza.get_ravno().size
    #Kreiranje pocetne populacije jedinki
    pomocna = []
    for i in range(args.popsize):
        jedinka = np.random.normal(0, 0.01, br_paramatara)
        pomocna.append(jedinka)
    populacija = np.array(pomocna)

    #Glavna petlja genetskog algoritma
    for e in range(1, args.iter + 1):
        pogreske = []
        for i in range(len(populacija)):
            kromosom = populacija[i]
            mreza.set_ravno(kromosom)
            predy = mreza.prolaz(X_train)
            pogreska = mse(y_train, predy)
            pogreske.append(pogreska)
            
        pogreske = np.array(pogreske)
        #Izracun dobrote
        dobrota = []
        for i in range(len(pogreske)):
            vrijednost = 1.0 / (pogreske[i] + 1e-8)
            dobrota.append(vrijednost)
        dobrota = np.array(dobrota)
        #Sortiranje indeksa jedinke od najbolje prema najlosijoj
        poredak = np.argsort(pogreske)
        
        if e % 2000 == 0:
            najbolji = poredak[0]
            print(f"[Train error @{e}]: {pogreske[najbolji]:.6f}")
            
        nova_gen = []
        #Elitizam
        for i in range(args.elitism):
            el_indeks = poredak[i]
            nova_gen.append(populacija[el_indeks])
            
        ukupna_dobrota = np.sum(dobrota)
        v = []
        for i in range(len(dobrota)):
            vjerojatnost = dobrota[i] / ukupna_dobrota
            v.append(vjerojatnost)
        v = np.array(v)
        #Popunjavanje ostatka nove generacije
        while len(nova_gen) < args.popsize:
            roditelj1 = selekcija(populacija, v)
            roditelj2 = selekcija(populacija, v)
            
            dijete = krizanje(roditelj1, roditelj2)
            dijete = mutacija(dijete, args.p, args.K)
            
            nova_gen.append(dijete)
            
        populacija = np.array(nova_gen)
    #Pronalazak konacnog pobjednika
    kraj_pogreske = []
    for i in range(len(populacija)):
        kromosom = populacija[i]
        mreza.set_ravno(kromosom)
        predy = mreza.prolaz(X_train)
        kraj_pogreske.append(mse(y_train, predy))
        
    kraj_pogreske = np.array(kraj_pogreske)
    najbolji_kraj = np.argmin(kraj_pogreske)
    najbolji_kromosom = populacija[najbolji_kraj]
    #Testiranje
    mreza.set_ravno(najbolji_kromosom)
    test_predikcije = mreza.prolaz(X_test)
    test_pogreska = mse(y_test, test_predikcije)

    print(f"[Test error]: {test_pogreska:.6f}")


if __name__ == "__main__":
    main()
