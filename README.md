# deep_learning
import numpy as np

class DeepNeuralNet():
    def __init__(self,X_train,y_train,neuron_per_layer_list,learning_rate=0.05,n_iteration=10000):
        self.X = X_train
        self.Y = y_train
        
        # parameters to be optimize
        self.W = None
        self.b = None
        
        # hyperparameters to be tuned
        self.l_list = neuron_per_layer_list
        
        self.n = [self.X.shape[0]]+neuron_per_layer_list
        self.L = len(self.n)-1
        
        self.learning_rate = learning_rate
        self.n_iteration   = n_iteration
        
        
        self.costs = []
        
        # no of training examples
        self.m = self.X.shape[-1]
        '''
        n = [X.shape[0],12,1]
        L=len(n)-1
        learning_rate=0.8
        n_iteration = 10000
        '''
        self.__w_b_Initializer__()
        self.__propgate__()
        
    def __w_b_Initializer__(self): 
        # W_b initialization
        self.W,self.b = [[None],[None]] 
        L,n = self.L,self.n
        
        for i in range(1,L+1):
            w_i = np.random.randn(n[i],n[i-1])*np.sqrt(1/n[i-1])
            b_i = np.zeros((n[i],1))

            self.W.append(w_i)
            self.b.append(b_i)    

    def __propgate__(self):
        costs = self.costs
        for iterations in range(self.n_iteration):
            # forward propogation
            L,m,X,Y,W,b,n = self.L,self.m,self.X,self.Y,self.W,self.b,self.n
            
            A = [X]
            Z = [None]
            
            for i in range(1,L+1):
                z_i = np.dot(W[i],A[i-1])+b[i]

                if i==L:
                    a_i = sigmoid(z_i)
                else:
                    a_i = np.tanh(z_i)

                assert(a_i.shape == (n[i],m))
                assert(z_i.shape == (n[i],m))

                A.append(a_i)
                Z.append(z_i)

            # computing cost
            if iterations%1000 == 0:
                cost = (-1/m)*(np.dot(np.log(A[L]),Y.T)+np.dot(np.log(1-A[L]),(1-Y).T))
                cost = np.squeeze(cost)
                costs.append(cost)
                print(f"cost after the iteration {iterations}:{cost}")
            # backward propagation
            dZ,dW,db = [[None],[None],[None]]

            for j in range(0,L):
                i = L-j
                # print(f"i:{i} and j:{j}")
                if i==L:
                    dZ_i = A[i]-Y
                else:
                    dZ_i = np.dot(W[i+1].T,dZ[j])*(1-np.power(A[i],2))

                assert(dZ_i.shape == (n[i],m))
                dZ.append(dZ_i)

                dW_i = (1/m)*np.dot(dZ[j+1],A[i-1].T)
                db_i = (1/m)*np.sum(dZ[j+1],axis=1,keepdims=True)   


                assert(dW_i.shape == (n[i],n[i-1]))
                assert(db_i.shape == (n[i],1))


                dW.append(dW_i)
                db.append(db_i)

            dZ = dZ[:1]+dZ[1:][::-1]
            dW = dW[:1]+dW[1:][::-1]
            db = db[:1]+db[1:][::-1]
            for i in range(1,L+1):
                W[i] = W[i]-learning_rate*dW[i]
                b[i] = b[i]-learning_rate*db[i]

                assert(W[i].shape == (n[i],n[i-1]))
                assert(b[i].shape == (n[i],1))  
                
        self.W = W
        self.b = b
        self.costs = costs
        
    def predict(self,X_test):
        X,W,b = X_test,self.W,self.b
        L,m,n = self.L,X_test.shape[-1],self.n
        
        A = [X]
        Z = [None]

        for i in range(1,L+1):
            z_i = np.dot(W[i],A[i-1])+b[i]

            if i==L:
                a_i = sigmoid(z_i)
            else:
                a_i = np.tanh(z_i)

            assert(a_i.shape == (n[i],m))
            assert(z_i.shape == (n[i],m))

            A.append(a_i)
            Z.append(z_i)
        
        predictions = A[-1]>0.5
        
        return(predictions)
    
    def accuracy(self,X_test,y_test):
        predictions = self.predict(X_test)
        Y = y_test
        print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
        return float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
