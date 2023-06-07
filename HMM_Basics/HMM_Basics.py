import numpy as np

class ProbHiddenPath:
    def __init__(self):
        path, transition = self.readFromFile()
        P = self.calculateProb(path, transition)
        print(P)
        f = open('result.txt', 'w')
        f.write(str(P))
        f.close()

    def readFromFile(self):
        with open(r'C:\Users\prav0\source\repos\HMM_Basics\input.txt', 'r') as file:
            f = file.read()
        data = f.split()
        path = data[0]
        transition = {'A':{'A':float(data[-5]), 'B':float(data[-4])}, 'B':{'A':float(data[-2]), 'B':float(data[-1])}}
        return path, transition

    def calculateProb(self, path, transition):
        P = 0.5
        for i in range(len(path)-1):
            P *= transition[path[i]][path[i+1]]
        return P        
    
#ProbHiddenPath()

class ProbOutcomeGivenPath:
    def __init__(self):
        x, path, emission = self.readFromFile()
        P = self.calculatePrxpi(x, path, emission)
        print(P)
        f = open('result.txt', 'w')
        f.write(str(P))
        f.close()

    def readFromFile(self):
        with open(r'C:\Users\prav0\source\repos\HMM_Basics\input.txt', 'r') as file:
            f = file.read()
        data = f.split()
        x = data[0]
        path = data[6]
        emission = {'A':{'x':float(data[-7]), 'y':float(data[-6]), 'z':float(data[-5])}, 'B':{'x':float(data[-3]), 'y':float(data[-2]), 'z':float(data[-1])}}
        return x, path, emission

    def calculatePrxpi(self, x, path, emission):
        P = 1
        for i in range(len(x)):
            P *= emission[path[i]][x[i]]
        return P
    
#ProbOutcomeGivenPath()

class Decoding:
    def __init__(self):
        x, transionLog, emissionLog, stateDict = self.readFromFile()
        path = self.viterbi(x, transionLog, emissionLog, stateDict)
        print(path)
        f = open('result.txt', 'w')
        f.write(path)
        f.close()

    def readFromFile(self):
        with open(r'C:\Users\prav0\source\repos\HMM_Basics\input.txt', 'r') as file:
            f = file.read()
        data = f.split()
        x = data[0]
        ind = [i for i in range(len(data)) if '--------' == data[i]]
        alphabet = data[ind[0]+1:ind[1]]
        states = data[ind[1]+1:ind[2]]
        stateDict = {i:states[i] for i in range(len(states))}
        transitionLog = {i:{k:np.log10(float(data[ind[2]+len(states)+2+i*(len(states)+1)+k])) for k in range(len(states))} for i in range(len(states))}
        emissionLog = {i:{alphabet[k]:np.log10(float(data[ind[3]+len(alphabet)+2+i*(len(alphabet)+1)+k])) for k in range(len(alphabet))} for i in range(len(states))}
        return x, transitionLog, emissionLog, stateDict

    def viterbi(self, x, transionLog, emissionLog, stateDict):
        n = len(x)
        l = len(transionLog)
        s = [[0 for _ in range(l)] for __ in range(n)]
        backTrack = [[0 for _ in range(l)] for __ in range(n)]
        for k in range(l):
            s[0][k] = np.log10(1/l) + emissionLog[k][x[0]]
        for i in range(1, n):
            for k in range(l):
                currS = [s[i-1][kpre] + transionLog[kpre][k] + emissionLog[k][x[i]] for kpre in range(l)]
                ind = np.argmax(currS)
                backTrack[i][k] = ind
                s[i][k] = currS[ind]
        
        currState = np.argmax(s[n-1])
        stateList = [currState]
        for i in range(n-1, 0, -1):
            currState = backTrack[i][currState]
            stateList.insert(0, currState)
        path = ''.join([stateDict[state] for state in stateList])
        return path

#Decoding()


class OutcomeLikelihood:
    def __init__(self):
        x, transition, emission = self.readFromFile()
        prob = self.calculatePrx(x, transition, emission)
        print(prob)
    
    def readFromFile(self):
        with open(r'C:\Users\prav0\source\repos\HMM_Basics\input.txt', 'r') as file:
            f = file.read()
        data = f.split()
        x = data[0]
        ind = [i for i in range(len(data)) if '--------' == data[i]]
        alphabet = data[ind[0]+1:ind[1]]
        states = data[ind[1]+1:ind[2]]
        stateDict = {i:states[i] for i in range(len(states))}
        transition = {i:{k:float(data[ind[2]+len(states)+2+i*(len(states)+1)+k]) for k in range(len(states))} for i in range(len(states))}
        emission = {i:{alphabet[k]:float(data[ind[3]+len(alphabet)+2+i*(len(alphabet)+1)+k]) for k in range(len(alphabet))} for i in range(len(states))}
        return x, transition, emission
    
    def calculatePrx(self, x, transition, emission):
        n = len(x)
        l = len(transition)
        forward = [[0 for _ in range(l)] for __ in range(n)]
        for k in range(l):
            forward[0][k] = 1/l*emission[k][x[0]]
        for i in range(1, n):
            for k in range(l):
                forward[i][k] = sum([forward[i-1][kpre]*transition[kpre][k]*emission[k][x[i]] for kpre in range(l)])
        return sum(forward[n-1])

#OutcomeLikelihood()


class HMMParameterEstimation:
    def __init__(self):
        x, path, alphabet, states = self.readFromFile()
        transition, emission = self.estimateParameters(x, path, alphabet, states)
        #print(transition, emission)
        self.saveParameters(transition, emission, alphabet, states)

    def readFromFile(self):
        with open(r'C:\Users\prav0\source\repos\HMM_Basics\input.txt', 'r') as file:
            f = file.read()
        data = f.split()
        ind = [i for i in range(len(data)) if '--------' == data[i]]
        x = data[0]
        alphabet = data[ind[0]+1:ind[1]]
        states = data[ind[2]+1:]
        path = data[ind[1]+1]
        return x, path, alphabet, states

    def estimateParameters(self, x, path, alphabet, states):
        alphabet2ind = {alphabet[i]:i for i in range(len(alphabet))}
        states2ind = {states[i]:i for i in range(len(states))}
        transition = np.zeros((len(states), len(states)), dtype = float)
        emission = np.zeros((len(states), len(alphabet)), dtype = float)
        
        for i in range(len(path)-1):
            transition[states2ind[path[i]], states2ind[path[i+1]]] += 1
        
        for i in range(len(x)):
            emission[states2ind[path[i]], alphabet2ind[x[i]]] += 1
        
        for i in range(len(states)):
            sum1 = sum(transition[i,:])
            if 0 == sum1:
                transition[i,:] += 1/len(states)
            else:
                transition[i,:] /= sum1
            sum2 = sum(emission[i,:])
            if 0 == sum2:
                emission[i,:] += 1/len(alphabet)
            else:
                emission[i,:] /= sum2
        
        return transition, emission

    def saveParameters(self, transition, emission, alphabet, states):
        f = open('result.txt', 'w')
        print(' '.join([' ']+states))
        f.write('\t'+'\t'.join(states)+'\n')
        for i in range(len(states)):
            print(' '.join([states[i]]+list(map(str, transition[i, :]))))
            f.write('\t'.join([states[i]]+list(map(str, transition[i, :])))+'\n')
        print('--------')
        f.write('--------\n')
        print(' '.join([' ']+alphabet))
        f.write('\t'+'\t'.join(alphabet)+'\n')
        for i in range(len(states)):
            print(' '.join([states[i]]+list(map(str, emission[i, :]))))
            f.write('\t'.join([states[i]]+list(map(str, emission[i, :])))+'\n')
        f.close()

#HMMParameterEstimation()


class SoftDecoding:
    def __init__(self):
        x, transition, emission, alphabet, states = self.readFromFile()
        Pr = self.softDecode(x, transition, emission, alphabet, states)
        self.savePr(Pr, states)

    def readFromFile(self):
        with open(r'C:\Users\prav0\source\repos\HMM_Basics\input.txt', 'r') as file:
            f = file.read()
        data = f.split()
        x = data[0]
        ind = [i for i in range(len(data)) if '--------' == data[i]]
        alphabet = data[ind[0]+1:ind[1]]
        states = data[ind[1]+1:ind[2]]
        transition = np.zeros((len(states), len(states)), dtype = float)
        emission = np.zeros((len(states), len(alphabet)), dtype = float)
        for i in range(len(states)):
            transition[i, :] = [float(d) for d in data[ind[2]+len(states)+2+i*(len(states)+1):ind[2]+len(states)+1+(i+1)*(len(states)+1)]]
            emission[i, :] = [float(d) for d in data[ind[3]+len(alphabet)+2+i*(len(alphabet)+1):ind[3]+len(alphabet)+1+(i+1)*(len(alphabet)+1)]]
        return x, transition, emission, alphabet, states

    def softDecode(self, x, transition, emission, alphabet, states):
        n = len(x)
        l = transition.shape[0]
        x2ind = {alphabet[i]:i for i in range(len(alphabet))}
        xList = [x2ind[x[i]] for i in range(len(x))]
        forward = [[0 for _ in range(l)] for __ in range(n)]
        backward = [[0 for _ in range(l)] for __ in range(n)]
        for k in range(l):
            forward[0][k] = emission[k, xList[0]]/l
        for i in range(1, n):
            for k in range(l):
                forward[i][k] = sum([forward[i-1][kpre]*transition[kpre, k]*emission[k, xList[i]] for kpre in range(l)])
        fsink = sum(forward[n-1])

        for k in range(l):
            backward[n-1][k] = 1
        for i in range(n-2, -1, -1):
            for k in range(l):
                backward[i][k] = sum([backward[i+1][kpre]*transition[k, kpre]*emission[kpre, xList[i+1]] for kpre in range(l)])
        
        Pr = np.zeros((n, l), dtype = float)
        for i in range(n):
            for k in range(l):
                Pr[i, k] = forward[i][k]*backward[i][k]/fsink

        return Pr

    def savePr(self, Pr, states):
        f = open('result.txt', 'w')
        print(' '.join(states))
        f.write('\t'.join(states)+'\n')
        for i in range(Pr.shape[0]):
            print(' '.join(list(map(str, Pr[i, :]))))
            f.write('\t'.join(list(map(str, Pr[i, :])))+'\n')
        f.close()

#SoftDecoding()

class BaumWelch:
    def __init__(self):
        x, transition, emission, alphabet, states, iterNo = self.readFromFile()
        transition, emission = self.BaumWelchLearning(x, transition, emission, alphabet, states, iterNo)
        self.saveTransitionAndEmission(alphabet, states, transition, emission)

    def readFromFile(self):
        with open(r'C:\Users\prav0\source\repos\HMM_Basics\input.txt', 'r') as file:
            f = file.read()
        data = f.split()
        iterNo = int(data[0])
        x = data[2]
        ind = [i for i in range(len(data)) if '--------' == data[i]]
        alphabet = data[ind[1]+1:ind[2]]
        states = data[ind[2]+1:ind[3]]
        transition = np.zeros((len(states), len(states)), dtype = float)
        emission = np.zeros((len(states), len(alphabet)), dtype = float)
        for i in range(len(states)):
            transition[i, :] = [float(d) for d in data[ind[3]+len(states)+2+i*(len(states)+1):ind[3]+len(states)+1+(i+1)*(len(states)+1)]]
            emission[i, :] = [float(d) for d in data[ind[4]+len(alphabet)+2+i*(len(alphabet)+1):ind[4]+len(alphabet)+1+(i+1)*(len(alphabet)+1)]]
        return x, transition, emission, alphabet, states, iterNo

    def softDecode(self, xList, transition, emission):
        n = len(xList)
        l = transition.shape[0]
        forward = [[0 for _ in range(l)] for __ in range(n)]
        backward = [[0 for _ in range(l)] for __ in range(n)]
        for k in range(l):
            forward[0][k] = emission[k, xList[0]]/l
        for i in range(1, n):
            for k in range(l):
                forward[i][k] = sum([forward[i-1][kpre]*transition[kpre, k]*emission[k, xList[i]] for kpre in range(l)])
        fsink = sum(forward[n-1])

        for k in range(l):
            backward[n-1][k] = 1
        for i in range(n-2, -1, -1):
            for k in range(l):
                backward[i][k] = sum([backward[i+1][kpre]*transition[k, kpre]*emission[kpre, xList[i+1]] for kpre in range(l)])
        
        Pr = np.zeros((l, n), dtype = float)
        for i in range(n):
            for k in range(l):
                Pr[k, i] = forward[i][k]*backward[i][k]/fsink
        
        Pr2 = np.zeros((l, l, n-1), dtype = float)
        for k1 in range(l):
            for k2 in range(l):
                for i in range(n-1):
                    Pr2[k1, k2, i] = forward[i][k1]*transition[k1, k2]*emission[k2, xList[i+1]]*\
                    backward[i+1][k2]/fsink

        return Pr, Pr2
    
    def estimateParameters(self, xList, Pr, Pr2, nAlphabet):
        n = len(xList)
        l = Pr2.shape[0]
        transition = np.zeros((l, l), dtype = float)
        emission = np.zeros((l, nAlphabet), dtype = float)
        for k1 in range(l):
            for k2 in range(l):
                transition[k1, k2] = sum(Pr2[k1, k2, :])
        for k in range(l):
            for i in range(n):
                emission[k, xList[i]] += Pr[k, i]

        for i in range(l):
            sum1 = sum(transition[i,:])
            if 0 == sum1:
                transition[i,:] += 1/l
            else:
                transition[i,:] /= sum1
            sum2 = sum(emission[i,:])
            if 0 == sum2:
                emission[i,:] += 1/nAlphabet
            else:
                emission[i,:] /= sum2
        return transition, emission

    def BaumWelchLearning(self, x, transition, emission, alphabet, states, iterNo):
        x2ind = {alphabet[i]:i for i in range(len(alphabet))}
        xList = [x2ind[x[i]] for i in range(len(x))]
        for _ in range(iterNo):
            Pr, Pr2 = self.softDecode(xList, transition, emission)
            transition, emission = self.estimateParameters(xList, Pr, Pr2, len(alphabet))
        return transition, emission

    def saveTransitionAndEmission(self, alphabet, states, fullTransition, emission):
        f = open('result.txt', 'w')
        print(' '.join([' '] + states))
        f.write('\t'+'\t'.join(states) + '\n')
        for i in range(fullTransition.shape[0]):
            print(' '.join([states[i]] + ['{:.3f}'.format(a) for a in fullTransition[i, :]]))
            f.write('\t'.join([states[i]] + ['{:.3f}'.format(a) for a in fullTransition[i, :]]) + '\n')
        print('--------')
        f.write('--------'+'\n')
        print(' '.join([' '] + alphabet))
        f.write('\t'+'\t'.join(alphabet)+'\n')
        for i in range(emission.shape[0]):
            print(' '.join([states[i]] + ['{:.3f}'.format(a) for a in emission[i, :]]))
            f.write('\t'.join([states[i]] + ['{:.3f}'.format(a) for a in emission[i, :]])+'\n')
        f.close()

#BaumWelch()