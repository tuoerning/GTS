#Shen et al. Multiple-boundary clustering and prioritization to promote neural network retraining
#https://github.com/actionabletest/MCP


import numpy as np


def GetMcp(budget_ratio_list,nb_class,model,x_test):
    
    pred_test_prob = model.predict(x_test)
    top_list = []
    ans = []
    size = len(pred_test_prob)
    for ratio_ in budget_ratio_list:
        top_list.append(int(size*ratio_))
    for budget in top_list:
        
        dicratio=[[] for i in range(nb_class*nb_class)]
        dicindex=[[] for i in range(nb_class*nb_class)]
        for i in range(len(pred_test_prob)):
            act=pred_test_prob[i]
            max_index,sec_index,ratio = find_second(act)
            dicratio[max_index*nb_class+sec_index].append(ratio)
            dicindex[max_index*nb_class+sec_index].append(i)
        selected = select_from_firstsec_dic(budget, dicratio, dicindex, num_classes=nb_class)
        
        ans.append(selected)

    return ans


def find_second(output_probability):
    output = np.array(output_probability)
    sorted_index = output.argsort()[::-1] 
    max_value = output[sorted_index[0]]
    second_max_value = output[sorted_index[1]]
    bound_priority =  1.0* second_max_value / max_value
    return sorted_index[0], sorted_index[1], bound_priority


def select_from_firstsec_dic(selectsize, dicratio, dicindex, num_classes):
    selected_lst=[]
    tmpsize=selectsize
    
    
    noempty=no_empty_number(dicratio)
    print(selectsize)
    print(noempty)
   
    while selectsize>=noempty:
        for i in range(num_classes*num_classes):
            if len(dicratio[i])!=0:
                tmp=max(dicratio[i])
                j = dicratio[i].index(tmp)
                #if tmp>=0.1:
                selected_lst.append(dicindex[i][j])
                dicratio[i].remove(tmp)
                dicindex[i].remove(dicindex[i][j])
        selectsize=tmpsize-len(selected_lst)
        noempty=no_empty_number(dicratio)
        print(selectsize)
        
    
    
    while len(selected_lst)!= tmpsize:
        max_tmp=[0 for i in range(selectsize)]
        max_index_tmp=[0 for i in range(selectsize)]
        for i in range(num_classes*num_classes):
            if len(dicratio[i])!=0:
                tmp_max=max(dicratio[i])
                if tmp_max>min(max_tmp):
                    
                    index=max_tmp.index(min(max_tmp))
                    max_tmp[index]=tmp_max
                    #selected_lst.append()
                    #if tmp_max>=0.1:
                    max_index_tmp[index]=dicindex[i][dicratio[i].index(tmp_max)]
        if len(max_index_tmp)==0 and len(selected_lst)!= tmpsize:
            print('wrong!!!!!!')  
            break
        selected_lst=selected_lst+ max_index_tmp
        print(len(selected_lst))
    #print(selected_lst)
    assert len(selected_lst)== tmpsize
    return selected_lst


def no_empty_number(dicratio):
    no_empty=0
    for i in range(len(dicratio)):
        if len(dicratio[i])!=0:
            no_empty+=1
    return no_empty