# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:46:21 2023

@author: hhelmick
"""

import glob
import pickle
import time as count_time

import numpy as np
import pandas as pd

import cv2

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

from PIES_functions import weighting
from PIES_functions import get_lab


# [READ IN THE MODESLS THAT WERE SELECTED IN GRID SEARCHING]

name = r'PATH\TO\THIS\FILE\all_scaler.pkl'
with open(name, 'rb') as file:
    all_scaler = pickle.load(file)

name = r'PATH\TO\THIS\FILE\light_scaler.pkl'
with open(name, 'rb') as file:
    light_scaler = pickle.load(file)

name = r'PATH\TO\THIS\FILE\dark_scaler.pkl'
with open(name, 'rb') as file:
    dark_scaler = pickle.load(file)

name = r'PATH\TO\THIS\FILE\ann_all_data.pkl'
with open(name, 'rb') as file:
    all_ann = pickle.load(file)

name = r'PATH\TO\THIS\FILE\ann_light_data.pkl'
with open(name, 'rb') as file:
    light_ann = pickle.load(file)

name = r'PATH\TO\THIS\FILE\ann_dark_data.pkl'
with open(name, 'rb') as file:
    dark_ann = pickle.load(file)
        
# In[]

def get_files(path_in):
    
    path = path_in
    files = glob.glob(path + '\*.png')
    
    names = []
    for f in files:
        t1 = f.split('\\')
        names.append(t1[-1])
        
    return [files, names]

t15_back = get_files(r'PATH\TO\BACKGROUNDS')
t15_pie = get_files(r'PATH\TO\CROPPED_IMAGES')

t30_back = get_files(r'PATH\TO\BACKGROUNDS')
t30_pie = get_files(r'PATH\TO\CROPPED_IMAGES')

all_back = t15_back[0] + t30_back[0]
all_pie = t15_pie[0] + t30_pie[0]

all_pie_names = t15_pie[1] + t30_pie[1]

# In[]

def back_weights(files_in):
    
    back_weights = []
    
    for path in files_in:
        back = cv2.imread(path)
        weights = weighting(back, False)
        back_weights.append(weights)    

    return back_weights

start_time = count_time.time()

t15_back_weights = back_weights(t15_back[0])
t30_back_weights = back_weights(t30_back[0])

all_back_weights = t15_back_weights + t30_back_weights

elap = count_time.time() - start_time

print('background processing time s/img')
print(elap / len(all_pie))

'''
background processing time s/img
1.7902176521326367
'''

with open(r'PATH\TO\THIS\FILE\back_data_list.txt', 'w') as f:
    f.write(str(all_back_weights))

# In[]

# didn't feel like figuring out how to parase the file. Just coppied the list from the read data
back_weights_f = open(r'PATH\TO\THIS\FILE\back_data_list.txt', 'r').read()
all_back_weights = [[0.7867130659420453, 0.17427671650344279, 0.816822619230973, 0.18313416843664165, -1.8152420298067253, -0.6592337663146707], [0.7867498627756945, 0.17435116869379963, 0.8167708139215573, 0.18318599735859242, -1.81494511991654, -0.6583252564705092], [0.7650727747003003, 0.17327509791612028, 0.8124371709776099, 0.1875215663228702, -1.7901081481191152, -0.8435112996499685], [0.7360428386137869, 0.17584180150315454, 0.8026965964605571, 0.1972661398740443, -1.7342848401672317, -1.0374767915172158], [0.7451274500371501, 0.17483366283644564, 0.8059974033165344, 0.19396402622773912, -1.7532015164451658, -0.9799653469119209], [0.7647787543438133, 0.1732254926350819, 0.8124040444997255, 0.18755470715732314, -1.7899182957410782, -0.8460525564796006], [0.7488979788151785, 0.17417564921683248, 0.8075394762700594, 0.19242132619881558, -1.7620391143631466, -0.9569681461329084], [0.7564878031715714, 0.17397149698986736, 0.809690415768229, 0.19026949377302682, -1.7743662209060402, -0.9034000499359942], [0.7680017304832412, 0.17399488169103938, 0.8125796738436182, 0.1873790016361192, -1.7909248518958247, -0.8164729543832345], [0.7522530056491485, 0.17346668596898474, 0.8090010522723445, 0.19095914579895923, -1.7704154422482417, -0.9369947258675959], [0.7606070313165922, 0.1738679606084147, 0.8108292309735872, 0.18913019697835975, -1.780892859040275, -0.873553585415967], [0.7549117308949288, 0.16897534699176486, 0.813364755913845, 0.18659357716983038, -1.7954242768566195, -0.9430978287467268], [0.7506065431859281, 0.17382337693928374, 0.8082779447915494, 0.1916825535399329, -1.7662712885602576, -0.9467699701716217], [0.754274202164831, 0.1801127263639194, 0.8041370804273329, 0.1958250914628924, -1.742540112633277, -0.8831647409535943], [0.7299146278516456, 0.17650179762203422, 0.800441812390327, 0.19952178979497004, -1.721362993346876, -1.0755776860205848], [0.7304551985349397, 0.17654624998664148, 0.8005586723708277, 0.1994048854600876, -1.722032698337473, -1.0717153204746124], [0.7190870967119576, 0.178120162585223, 0.796002670851693, 0.20396257543442442, -1.6959232460902052, -1.139297938337309], [0.7216565464109879, 0.1782605001867088, 0.7966404705968653, 0.20332454426725333, -1.6995783096104389, -1.1217069312094077], [0.7940652117525228, 0.17564782481579622, 0.8173915246075558, 0.1825650028173963, -1.8185025818701284, -0.5788325218445368], [0.794226950330392, 0.17576685987441032, 0.8173319924656391, 0.1826245622728604, -1.8181613864501607, -0.5761730882829529], [0.71585952488383, 0.17874378930142765, 0.7945431932967906, 0.20542257666739394, -1.6875593951087, -1.1573285827047752], [0.7932395911365864, 0.17408051064571872, 0.8184711389760432, 0.18148488980801458, -1.8246901736910233, -0.6004450083283569], [0.7934282528271365, 0.17418954921715812, 0.8184255209337491, 0.1815305290476018, -1.8244287223664482, -0.5977070166780841], [0.7328917091772014, 0.17563574743271249, 0.8019854258368704, 0.1979775858950228, -1.7302092094696977, -1.0599573488025038], [0.7933194116038376, 0.17473492583427608, 0.8179601663270831, 0.18199609924435645, -1.8217616326525663, -0.5940890050794174], [0.7929777045897164, 0.17477342859693257, 0.8178518715343888, 0.18210444403984816, -1.8211409629952335, -0.597059123146866], [0.7376890770470486, 0.1753611134082136, 0.8035430745529185, 0.19641933117767663, -1.739135917056766, -1.0287357599160636], [0.7931274219529509, 0.1758934262155445, 0.816981675213435, 0.18297503986845975, -1.8161536220921088, -0.5859127222080095], [0.793401414144054, 0.17615956744567207, 0.8168290704887505, 0.18312771423742502, -1.8152790036719975, -0.5808740936489897], [0.7540880230947482, 0.17320348652803397, 0.8096945047857174, 0.19026540303740525, -1.7743896553193967, -0.925282553752795], [0.7907410963864477, 0.17444469390669826, 0.8176102559555607, 0.18234617095165395, -1.8197561935618425, -0.6210492939698387], [0.7909929081635801, 0.17442513297533013, 0.8176833879906615, 0.18227300525151968, -1.8201753344215186, -0.6188524578196448], [0.7595816372501781, 0.173226046828597, 0.8110909603305985, 0.18886835605977348, -1.7823928552308852, -0.8850044007939956], [0.7915732411586194, 0.17453767537595377, 0.8177242619476065, 0.18223211246644488, -1.8204095949698618, -0.6124556299740429], [0.7914076175956454, 0.17481112964291157, 0.8174655667782695, 0.18249092664941036, -1.8189269384890596, -0.6117392214654968], [0.7603101283369011, 0.17346585439122975, 0.8110811342235451, 0.18887818636116793, -1.782336540829715, -0.8781739195857112], [0.7925355722831149, 0.1744621787694789, 0.818003433867938, 0.18195281170793287, -1.8220096118947344, -0.6039472365092333], [0.792743973664324, 0.1744235007888092, 0.8180818435397753, 0.18187436577422, -1.822459001366414, -0.6022824986080978], [0.7169506426296919, 0.17840398263619062, 0.795142369123589, 0.20482318683307676, -1.6909930959021886, -1.151874073483943], [0.7938570349554167, 0.17707846280373063, 0.8161920569201663, 0.18376501718945637, -1.8116281163761954, -0.5681039029444694], [0.7942463347070341, 0.17746139248048143, 0.8159720204612727, 0.18398515311063357, -1.810367034098448, -0.5606603928392435], [0.7916594081119007, 0.17391100861655473, 0.8182507854844896, 0.18170534558628115, -1.8234272592328638, -0.6168456450953731], [0.793204028333546, 0.17415284892785576, 0.8184046115818245, 0.18155144811166812, -1.8243088843264363, -0.6001719165814428], [0.727387963248286, 0.17678499565459138, 0.7994926848674466, 0.2004712754856358, -1.7159237140636785, -1.0910594604553732], [0.793199377033059, 0.1751937342071176, 0.8175625139072774, 0.18239393496163725, -1.819482570153052, -0.5912975790455244], [0.7933531136922871, 0.17519385248509278, 0.8175971489689269, 0.18235928396881873, -1.819681073645115, -0.5898002346997161], [0.7224481186847875, 0.17761420978663967, 0.7973955186718744, 0.20256922017749557, -1.7039053041932017, -1.1196938706644686], [0.7932280922700976, 0.17650225547153053, 0.8165137264361748, 0.18344320181263185, -1.8134716849302976, -0.5795566645976502], [0.7936499336715624, 0.17637130910177534, 0.8167146149349116, 0.1832422219442995, -1.8146230287945762, -0.5765312215039047], [0.7648151025382494, 0.17354999023959783, 0.8121489756079532, 0.18780988641104468, -1.7884564620436556, -0.843773099364263], [0.7907799573998556, 0.1747931943717259, 0.817337275099096, 0.1826192772164802, -1.8181916627013464, -0.6178088113524528], [0.7908116072707552, 0.17481643369014954, 0.8173256949781863, 0.1826308626485128, -1.8181252937877257, -0.6173187776532214], [0.7643999508448226, 0.17323662610380153, 0.8123001481437999, 0.18765864850535607, -1.7893228517063309, -0.8488583143028837], [0.7915638787704761, 0.17351462845052856, 0.8185500441947532, 0.18140594789771936, -1.8251424044276099, -0.6210007874689388], [0.7907805774400962, 0.17493926213232713, 0.8172193389522102, 0.1827372674202451, -1.8175157378291296, -0.6165920752483265], [0.7899880686981966, 0.17479231308277787, 0.8171573934773066, 0.18279924125854963, -1.817160711245376, -0.6252252834179883], [0.7903047351045704, 0.17482402325328994, 0.8172040309917934, 0.1827525823917282, -1.8174280036795731, -0.6220094308387154], [0.7399031725779643, 0.17462524252019418, 0.80475050802132, 0.19521142077104847, -1.746055621483545, -1.0174108418721286], [0.7902690054296061, 0.17347597925939728, 0.8182869880909078, 0.1816691261929606, -1.8236347475377632, -0.6333054611105865], [0.7903864492022538, 0.17344292576820097, 0.818340523214673, 0.18161556623227382, -1.8239415739201301, -0.6324911735628583], [0.7214384824017717, 0.17767629229550852, 0.7970512660463203, 0.20291359892479488, -1.701932475609124, -1.1260420613477815], [0.7909408964457786, 0.17340543319746615, 0.8184970473015307, 0.181458969438682, -1.8248386624682809, -0.6276782444617802], [0.7909883890973108, 0.17343741012657488, 0.8184819346469734, 0.1814740891190113, -1.824752047080394, -0.6269792132961963], [0.7178739327855661, 0.1781359236611092, 0.795632739470107, 0.20433264032301723, -1.6938032701587618, -1.147156672300051], [0.788824056905339, 0.17305858417132858, 0.8182953444729699, 0.18166076593508995, -1.8236826405468065, -0.6497723936397359], [0.7897733499980757, 0.17291749178194427, 0.8186267248460434, 0.18132923155674452, -1.825581885649929, -0.6422917536094278], [0.7211713247828933, 0.1773271272633683, 0.7972570346038614, 0.20270775503706628, -1.70311168515832, -1.1295233955122912], [0.7896542298926477, 0.174146386446289, 0.8176037539501396, 0.18235267594876636, -1.8197189286947906, -0.6335643329683569], [0.7898962194076666, 0.17363650199050196, 0.8180719329834736, 0.18188428091566244, -1.8224022009695724, -0.6354401992323054], [0.7893934412987916, 0.1742164852515139, 0.8174873893941819, 0.1824690940078002, -1.8190520101038803, -0.6353991733655252], [0.7903305291312939, 0.17426108592506762, 0.817665270076139, 0.1822911315089819, -1.8200714953943637, -0.6263796407939177], [0.7402282852010065, 0.174626328193355, 0.8048385369142446, 0.1951233568575258, -1.7465601094194667, -1.0151651252240386], [0.7909456924676056, 0.17371028982024495, 0.8182512047802322, 0.18170492609615474, -1.823429662345875, -0.6251556971478099], [0.7910053969242345, 0.17404120452914074, 0.8179968755909015, 0.18195937301644804, -1.821972024440102, -0.6218944396779892], [0.7908065445287504, 0.17462234590321224, 0.8174814726874964, 0.18247501343297823, -1.8190180997746268, -0.6189719780392129], [0.7909446896094458, 0.1742683717890442, 0.8177992373013488, 0.1821571025527663, -1.8208393006848753, -0.6205956674584788], [0.7378284581668555, 0.1747564915427272, 0.8040738111721342, 0.19588838569791878, -1.7421775216687365, -1.030959182920612], [0.7910374286543037, 0.17467453336899297, 0.8174918209139602, 0.182464660451791, -1.8190774084051464, -0.6163707865605907], [0.7911794313113034, 0.17476565959800283, 0.8174504476067666, 0.1825060527654324, -1.8188402862280675, -0.6142753848959883], [0.738585096727346, 0.17526981936109742, 0.8038641143048375, 0.19609816523293122, -1.7409757666502528, -1.023066624464718], [0.7743530817505615, 0.17474651186508106, 0.8135218490022191, 0.1864364151986294, -1.7963246033762377, -0.761052535976592], [0.776514316782009, 0.17365380965437183, 0.8149290956467273, 0.18502854592407758, -1.8043897986339632, -0.7508962227084756], [0.788565726690648, 0.18703172264524048, 0.807027987582314, 0.19293302407588342, -1.7591077732953018, -0.5285555127115785], [0.7679733443428248, 0.17357304653493633, 0.8129157288595126, 0.1870428004335064, -1.7928508322646168, -0.8193772912203204], [0.7602518621083698, 0.17335819969433164, 0.8111539943889, 0.18880529508383026, -1.7827541097791602, -0.8792492676237206], [0.7712004203985158, 0.17415133354168066, 0.8132389140036838, 0.18671947417031365, -1.7947030565234245, -0.7903518090095009], [0.7712859557755062, 0.17414533976672253, 0.8132646865998379, 0.18669369029804972, -1.794850763402298, -0.7897126794426217], [0.7727152636418794, 0.17436931001245093, 0.813431087423006, 0.18652721659048, -1.7958044336410632, -0.7768260464018374], [0.7718798328137966, 0.174268091972117, 0.8133099130502423, 0.18664844405197667, -1.7951099634830914, -0.7841795655930536], [0.7654065535187803, 0.174888322392728, 0.8112086229362853, 0.18875064319265988, -1.7830671915843765, -0.8308880176028715], [0.7684189601974756, 0.17370541044562715, 0.8129181622271764, 0.18704036600525897, -1.7928647782616267, -0.8150766807746497], [0.7721984599207379, 0.17429602942462807, 0.8133648615455522, 0.18659347149184702, -1.7954248822492003, -0.7814513980872715], [0.749889933802623, 0.17240567721181432, 0.8092451868902275, 0.19071490925557688, -1.7718145892090504, -0.9597411888401], [0.778111653237626, 0.17343422459819513, 0.8154890608056319, 0.1844683301898401, -1.8075990802768134, -0.7392805826340608], [0.779302605169359, 0.17405455855003715, 0.8152684176795283, 0.18468907224370235, -1.806334525622551, -0.7250084664282639], [0.7340476677404807, 0.17528398833531011, 0.8025952509174412, 0.1973675247937906, -1.7337040406811877, -1.0539502756319137], [0.7679226709809642, 0.1730102973132882, 0.8133611941765402, 0.18659714046746356, -1.7954038639580832, -0.8233292220059494], [0.7733381651268986, 0.17380610532431795, 0.8140398306515642, 0.18591820554122807, -1.7992932470264817, -0.7755907841651575], [0.7762375276391725, 0.17463260490333177, 0.8140680812798367, 0.18588994243839774, -1.7994551565512962, -0.7464137434780462], [0.7536830322495242, 0.1742831764449584, 0.808709539885017, 0.1912507795260131, -1.7687447731343018, -0.9220326442300188], [0.7873681180882293, 0.17565253679306625, 0.8158606548257974, 0.1840965689897085, -1.809728771196193, -0.6424542357849642], [0.7858993097513072, 0.17540294633196085, 0.8157227843870029, 0.18423450154026244, -1.808938603480302, -0.65775677041813], [0.7856634744717592, 0.17502236348188438, 0.8159760004452237, 0.1839811713298768, -1.8103898443383728, -0.6628349878809825], [0.7857361554827029, 0.1750212117759109, 0.8159937881971245, 0.1839633755464959, -1.8104917902045887, -0.6621941869229544], [0.7868693510098822, 0.17524979401065277, 0.8160711246081847, 0.18388600419743917, -1.8109350238071085, -0.6502014088746895], [0.7815932911998444, 0.17344276085694166, 0.8163060023167252, 0.18365102018672186, -1.8122811656271955, -0.7100726262658255], [0.7833971689801058, 0.17501326381703108, 0.8154555667451308, 0.18450183928401898, -1.8074071183011635, -0.6829437182396829], [0.7832012813431872, 0.17565053826785149, 0.814893952683031, 0.1850637045597261, -1.804188387000773, -0.6798410297352044], [0.7852551568146466, 0.17486602397062867, 0.8160077751398687, 0.18394938228724, -1.8105719527298416, -0.6676813540011433], [0.7820004663435233, 0.17410299779805305, 0.8158659140052924, 0.18409130743894597, -1.8097589127944869, -0.7017991163180315], [0.7875553349368581, 0.19207056615213114, 0.8028147969515751, 0.19714789340233874, -1.7349622337055945, -0.4865730897491013], [0.7818869908651442, 0.17435454023807817, 0.8156352732742094, 0.18432205202612328, -1.80843705712578, -0.7009281919865966], [0.7853560389123236, 0.17516245080909087, 0.8157912863777265, 0.18416596870122726, -1.8093312043332406, -0.6644952758514385], [0.779369418650391, 0.17379294711039384, 0.8154966106105375, 0.18446077699542995, -1.8076423499071983, -0.72630495412208], [0.7789407678947089, 0.17336605860397125, 0.8157415943211896, 0.18421568313788683, -1.8090464075774118, -0.732887758813807], [0.7818649625286832, 0.17422458981104194, 0.8157354673946666, 0.18422181282294037, -1.8090112927387907, -0.702069747497104], [0.784272697665649, 0.1744624538904589, 0.8161060780964348, 0.18385103490809407, -1.8111353507269758, -0.6794317728768311], [0.7873985450702856, 0.17728812444027175, 0.8145475529157944, 0.18541025846416648, -1.8022030998192142, -0.6289005114267389], [0.7834883232930389, 0.1745634400812489, 0.8158412262263028, 0.18411600634795, -1.8096174213164948, -0.6855296194684559], [0.7830753547663802, 0.17396452392293793, 0.8162301811718122, 0.1837268756788295, -1.811846615849035, -0.6935732417220714], [0.785998278182033, 0.17523048373485128, 0.8158852211505948, 0.1840719915871668, -1.8098695666056688, -0.6582150976636484], [0.7833729396482664, 0.17528047444926964, 0.8152335765476814, 0.18472392897390433, -1.8061348435050173, -0.6811379013220197], [0.7805859461646297, 0.17442660997469817, 0.8152703354432882, 0.18468715362117827, -1.8063455167455196, -0.7115112746005425], [0.7831828269628872, 0.17502102402978204, 0.8153991412750646, 0.1845582900672743, -1.8070837313468107, -0.6847592367229518], [0.7835674052698756, 0.17529094951257063, 0.8152705700802545, 0.18468691887914146, -1.8063468615012277, -0.679350894000298], [0.7821341484290301, 0.1749516693648655, 0.8152094333690206, 0.18474808295777112, -1.8059964737088303, -0.694392308543125], [0.7834492898204896, 0.17487415559622477, 0.8155804030126732, 0.18437694695459894, -1.8081225831271004, -0.6835358473164537], [0.7811541541260629, 0.17409986737427663, 0.8156693668874166, 0.18428794307823304, -1.8086324554531554, -0.7090479145576152], [0.7808036753392632, 0.17471290443605225, 0.8150896333004897, 0.18486793659791645, -1.8053098737706255, -0.7075781434765893], [0.7839715725559275, 0.17450191132530313, 0.8160039058967439, 0.18395325327781964, -1.8105497771684278, -0.6817716835712978], [0.7826207559117104, 0.174600266288516, 0.8156083261584054, 0.18434901125796677, -1.8082826170405568, -0.6927907163991563], [0.7730725003427507, 0.17374254093279662, 0.8140270487395296, 0.18593099309600847, -1.7992199915550984, -0.7781422223017566], [0.7794590407827175, 0.1740136299544961, 0.8153387295113776, 0.18461872891415754, -1.8067374981582007, -0.7239876410661124], [0.7764555052770766, 0.17334499023675276, 0.8151659493812997, 0.18479158639901905, -1.8057472575818216, -0.7534930680824697], [0.7845830221283046, 0.17457555575989625, 0.8160867194372394, 0.18387040231936802, -1.8110244015507941, -0.6758478470836794], [0.7844133560605049, 0.17534366960514858, 0.8154253497327799, 0.18453206985416248, -1.8072339378520057, -0.67148254433854], [0.7845672547777753, 0.17497005855392356, 0.8157635808181742, 0.18419368674029402, -1.8091724173061827, -0.6729806612846283], [0.7840074070786394, 0.17428314297770675, 0.8161895464238108, 0.18376752882205438, -1.8116137281049847, -0.683104608980037], [0.785566648452067, 0.17507807426786126, 0.8159084532638481, 0.1840487489949072, -1.810002715350102, -0.6632682726939396], [0.7829757796478876, 0.17531913947087463, 0.8151093125622426, 0.1848482485411721, -1.805422659820076, -0.6843248156723531], [0.7824683967075736, 0.17453623543019536, 0.815624490883749, 0.1843328392650464, -1.8083752607768528, -0.6945830194031641], [0.7840522073180611, 0.17496463811879903, 0.815647930014087, 0.18430938959418186, -1.8085095958048207, -0.6775641233908734], [0.7826628593656977, 0.17464789874826703, 0.8155795980690042, 0.18437775226001418, -1.8081179698117804, -0.6920722113035621], [0.7824164550350665, 0.17549727012990213, 0.814833974263186, 0.1851237097121914, -1.8038446382867221, -0.6878694249091118], [0.7809512896981479, 0.17470886849495015, 0.8151277328514086, 0.1848298200178602, -1.805528230439168, -0.7063471045021887], [0.7807092147643749, 0.17437698609084729, 0.8153396790206043, 0.18461777897940268, -1.8067429400049053, -0.710823112824794], [0.7836264548472107, 0.17554274618365373, 0.8150806200656507, 0.18487695386025405, -1.805258217001755, -0.6769169678385513], [0.7839857980730471, 0.17214514762518907, 0.817920111941476, 0.18203617213159518, -1.8215320690170635, -0.699112119740619], [0.7819232874049977, 0.175253169694785, 0.8149157055213163, 0.18504194202154423, -1.804313057036412, -0.6939777159850999], [0.7730784538469693, 0.1736887198296102, 0.8140722384516694, 0.1858857834305515, -1.7994789820664028, -0.7784525211021454], [0.775050076511828, 0.17329948628486447, 0.8148653122181604, 0.18509235779207256, -1.8040242425671928, -0.7652112770836912], [0.7777863769716156, 0.17327701408093832, 0.8155392939709629, 0.1844180744666314, -1.8078869778797564, -0.7430543531758309]]

# In[]

def get_pie_data(files_in):
    
    pie_data = []

    for path in files_in:
        cropped = cv2.imread(path)
        cropped_data = get_lab(cropped)
        pie_data.append(cropped_data)

    return pie_data

start_time = count_time.time()

t15_pie_weights = get_pie_data(t15_pie[0])
t30_pie_weights = get_pie_data(t30_pie[0])

all_pie_weights = t15_pie_weights + t30_pie_weights
elap = count_time.time() - start_time

print('pie processing time s/img')
print(elap / len(all_pie))
'''
pie processing time s/img
0.23423869986283152
'''

# In[]

light_out = []
dark_out = []
all_out = []
weighted_out = []
weighted2_out = []
pie_out = []

for data, weights in zip(all_pie_weights, all_back_weights):
    
    im_weights = pd.DataFrame(weights)
    im_weights.index = ['light_weight1', 'dark_weight1', 'light_weight2', 'dark_weight2', 'PC1', 'PC2']
    
    cropped_data = data
    
    idx = ['Ls_mean', 'Ls_max', 'Ls_min', 'Ls_std', 
     'As_mean', 'As_max', 'As_min', 'As_std',
     'Bs_mean', 'Bs_max', 'Bs_min', 'Bs_std',
     'Bog_mean', 'Gog_mean', 'Rog_mean',
     'mean_slope_h', 'mean_slope_v', 'above_h', 'mean_above_v']
    
    pie = pd.DataFrame(cropped_data)
    pie.index = idx
    pie = pie.transpose()
    
    pie['abs_slope_h'] = pie['mean_slope_h'].abs()
    pie['abs_slope_v'] = pie['mean_slope_v'].abs()
    
    pie['count_above_std1_h'] = pie['above_h']
    pie['count_above_std1_v'] = pie['mean_above_v']

    '''
    pie['count_above_std1_h'] = pie['above_h'] = 0
    pie['count_above_std1_v'] = pie['mean_above_v'] = 0
    pie['abs_slope_h'] = pie['mean_slope_h'] = 0
    pie['abs_slope_v'] = pie['mean_slope_v'] = 0
    '''
    
    pie_out.append(pie.values.tolist())
    
    med_in_light = pie[['Ls_mean', 'As_mean', 'Bs_mean', 'abs_slope_h', 'count_above_std1_h']]
    med_in_dark = pie[['Bog_mean', 'Gog_mean', 'Rog_mean', 'abs_slope_h', 'abs_slope_v']]
    med_in_all = pie[['Ls_mean', 'As_mean', 'Bs_mean', 'count_above_std1_v']]
        
    light_scale = light_scaler.transform(med_in_light)
    dark_scale = dark_scaler.transform(med_in_dark)
    all_scale = all_scaler.transform(med_in_all)
    
    light_pred = light_ann.predict(light_scale)
    dark_pred = dark_ann.predict(dark_scale)
    all_pred = all_ann.predict(all_scale)
    
    light_preds = pd.DataFrame(light_pred)
    light_weighted = light_preds * im_weights.loc['light_weight1'][0]
    
    dark_preds = pd.DataFrame(dark_pred)
    dark_weighted = dark_preds * im_weights.loc['dark_weight1'][0]
    
    weighted_pred = light_weighted + dark_weighted
    
    light_out.append(light_pred[0])
    dark_out.append(dark_pred[0])
    all_out.append(all_pred[0])
    weighted_out.append(weighted_pred.iloc[0])
    
    if im_weights.loc['PC2'][0] < 0:
        weighted2 = [light_pred[0][0], light_pred[0][1], dark_pred[0][2]]
        weighted2_out.append(weighted2)

# In[]

pie_df_out = pd.DataFrame(pie_out)

cols = ['Ls_mean', 'Ls_max', 'Ls_min', 'Ls_std', 
 'As_mean', 'As_max', 'As_min', 'As_std',
 'Bs_mean', 'Bs_max', 'Bs_min', 'Bs_std',
 'Bog_mean', 'Gog_mean', 'Rog_mean',
 'mean_slope_h', 'mean_slope_v', 'above_h', 'mean_above_v',
 'abs_slope_h', 'abs_slope_v', 'count_above_std1_h', 'count_above_std1_v',
 
 ]

pie_data = pd.DataFrame(pie_df_out[0].tolist())
pie_data.columns = cols
pie_data.index = all_pie_names


light = pd.DataFrame(light_out)
dark = pd.DataFrame(dark_out)
all_df = pd.DataFrame(all_out)
weighted = pd.DataFrame(weighted_out)
weighted2 = pd.DataFrame(weighted2_out)

weighted.columns = ['L_pred', 'A_pred', 'B_pred']
light.columns = ['L_pred', 'A_pred', 'B_pred']
dark.columns = ['L_pred', 'A_pred', 'B_pred']
all_df.columns = ['L_pred', 'A_pred', 'B_pred']
weighted2.columns = ['L_pred', 'A_pred', 'B_pred']

weighted.index = all_pie_names
light.index = all_pie_names
dark.index = all_pie_names
all_df.index = all_pie_names
weighted2.index = all_pie_names

weighted = weighted2

pea = []
weight = []
time = []
ph = []

for i in weighted.index:
    t1 = i.split('_')
    pea.append(t1[0].upper().replace('PEA', ''))
    weight.append(t1[3].replace('weight', ''))
    time.append(t1[2].replace('time', ''))
    ph.append(t1[1].replace('ph', ''))

weighted['pea'] = pea
weighted['weight'] = weight
weighted['time'] = time
weighted['ph'] = ph

weighted['pea_weight_time_ph'] = weighted['pea'] + '_' + weighted['weight'] + '_' + weighted['time'] + '_' + weighted['ph']
weighted['pea_weight_time_ph'] = weighted['pea_weight_time_ph'].apply(lambda x: str(x).replace('0_0_30_0', 'NONE_0_30_0'))
weighted['pea_weight_time_ph'] = weighted['pea_weight_time_ph'].apply(lambda x: str(x).replace('0_0_15_0', 'NONE_0_15_0'))

weighted['pea_weight_time_ph'].unique()


color = pd.read_csv(r'PATH\TO\THIS\FILE\hunter_data.csv')
color = color.loc[color['b*'] < 50]

pea = []
gly = []
weight = []
rep = []
for i in color['ID'].to_list():
    t1 = i.split('_')
    pea.append(t1[0].strip('pea'))
    gly.append(t1[1].strip('gly'))
    weight.append(t1[2].strip('g'))
    rep.append(t1[3].strip('rep'))

color['pea'] = pea
color['gly'] = gly
color['weight'] = weight
color['rep'] = rep
color['str_time'] = color['bake_time'].apply(lambda x: str(x))
color['str_ph'] = color['ph'].apply(lambda x: str(x))

color['pea_weight_time_ph'] = color['pea'] + '_' + color['weight'] + '_' + color['str_time'] + '_' + color['str_ph']

color = color.sort_values(by = 'ID')
color['pea_weight_time_ph'].unique()


gloss = pd.read_csv(r'PATH\TO\THIS\FILE\gloss.csv')
gloss = gloss.drop(['Unnamed: 13', 'Unnamed: 14'], axis = 1)
gloss = gloss[0:16]
gloss = gloss.drop(['Unnamed: 28', 'Unnamed: 39'], axis = 1)

gmean = gloss.mean()
gstd = gloss.std()

gstat = pd.DataFrame()
gstat['mean'] = gmean
gstat['std'] = gstd

gmelt = gloss.melt()

pea = []
ph = []
time = []
weight = []
for i in gmelt['variable'].to_list():
    t1 = i.split('_')
    pea.append(t1[0].upper().replace('PEA', ''))
    ph.append(t1[1].strip('ph'))
    time.append(t1[2].strip('time'))
    weight.append(t1[3].strip('weight'))

gmelt['pea'] = pea
gmelt['ph'] = ph
gmelt['time'] = time
gmelt['weight'] = weight

gmelt['pea_weight_time_ph'] = gmelt['pea'] + '_' + gmelt['weight'] + '_' + gmelt['time'] + '_' + gmelt['ph']

gmelt['pea_weight_time_ph'] = gmelt['pea_weight_time_ph'].apply(lambda x: str(x).replace('0_0.1_30_0', 'NONE_0_30_0'))
gmelt['pea_weight_time_ph'] = gmelt['pea_weight_time_ph'].apply(lambda x: str(x).replace('0_0.1_15_0', 'NONE_0_15_0'))
gmelt['pea_weight_time_ph'] = gmelt['pea_weight_time_ph'].apply(lambda x: str(x).replace('0_0_30_0', 'NONE_0_30_0'))
gmelt['pea_weight_time_ph'] = gmelt['pea_weight_time_ph'].apply(lambda x: str(x).replace('0_0_15_0', 'NONE_0_15_0'))

gmelt['pea_weight_time_ph'].unique()


cg = color.merge(gmelt, on = 'pea_weight_time_ph')

g = cg.groupby('pea_weight_time_ph')
m = g.mean()
s = g.std()

new_c = []
for c in s.columns:
    new_c.append(c + '_' + 'std')

s.columns = new_c

ms = pd.concat([m, s], axis = 1)

ms['merge'] = m.index

s['L*_std'].describe()
s['a*_std'].describe()
s['b*_std'].describe()


merge = ms.merge(weighted, on = 'pea_weight_time_ph')

merge['L_error_ann'] = np.sqrt((merge['L*'] - merge['L_pred'])**2)
merge['A_error_ann'] = np.sqrt((merge['a*'] - merge['A_pred'])**2)
merge['B_error_ann'] = np.sqrt((merge['b*'] - merge['B_pred'])**2)
merge['Total_error_ann'] = np.sqrt((merge['L*'] - merge['L_pred'])**2 + (merge['a*'] - merge['A_pred'])**2 + (merge['b*'] - merge['B_pred'])**2)

merge['L_error_ann'].describe()
merge['A_error_ann'].describe()
merge['B_error_ann'].describe()
merge['Total_error_ann'].describe()

print(merge['L_error_ann'].describe())
print(merge['A_error_ann'].describe())
print(merge['B_error_ann'].describe())
print(merge['Total_error_ann'].describe())

# In[]

pea = []
weight = []
time = []
ph = []

for i in pie_data.index:
    t1 = i.split('_')
    pea.append(t1[0].upper().replace('PEA', ''))
    weight.append(t1[3].replace('weight', ''))
    time.append(t1[2].replace('time', ''))
    ph.append(t1[1].replace('ph', ''))

pie_data['pea'] = pea
pie_data['weight'] = weight
pie_data['time'] = time
pie_data['ph'] = ph

pie_data['pea_weight_time_ph'] = pie_data['pea'] + '_' + pie_data['weight'] + '_' + pie_data['time'] + '_' + pie_data['ph']
pie_data['pea_weight_time_ph'] = pie_data['pea_weight_time_ph'].apply(lambda x: str(x).replace('0_0_30_0', 'NONE_0_30_0'))
pie_data['pea_weight_time_ph'] = pie_data['pea_weight_time_ph'].apply(lambda x: str(x).replace('0_0_15_0', 'NONE_0_15_0'))

pie_data['pea_weight_time_ph'].unique()

pie_merge = ms.merge(pie_data, on = 'pea_weight_time_ph')

pie_merge['L_error_dir'] = np.sqrt((pie_merge['L*'] - pie_merge['Ls_mean'])**2)
pie_merge['A_error_dir'] = np.sqrt((pie_merge['a*'] - pie_merge['As_mean'])**2)
pie_merge['B_error_dir'] = np.sqrt((pie_merge['b*'] - pie_merge['Bs_mean'])**2)

pie_merge['Total_error_dir'] = np.sqrt((pie_merge['L*'] - pie_merge['Ls_mean'])**2 + (pie_merge['a*'] - pie_merge['As_mean'])**2 + 
                                   (pie_merge['b*'] - pie_merge['Bs_mean'])**2)

pie_merge['L_error_dir'].describe()
pie_merge['A_error_dir'].describe()
pie_merge['B_error_dir'].describe()
pie_merge['Total_error_dir'].describe()

print(pie_merge['L_error_dir'].describe())
print(pie_merge['A_error_dir'].describe())
print(pie_merge['B_error_dir'].describe())
print(pie_merge['Total_error_dir'].describe())

# In[]

pea = []
weight = []
time = []
for i in merge['merge'].to_list():
    t1 = i.split('_')
    pea.append(t1[0])
    weight.append(t1[1])
    time.append(t1[2])

merge['pea'] = pea
merge['weight'] = weight
merge['time'] = time

# In[]
merge = merge.loc[merge['B_pred'] > 15]

model = 'insert_model_name'
plt.figure(0)
sns.lmplot(data = merge, x = 'L*', y = 'L_pred', hue = 'time')
plt.title(model)
plt.figure(1)
sns.lmplot(data = merge, x = 'a*', y = 'A_pred', hue = 'time')
plt.title(model)
plt.figure(2)
sns.lmplot(data = merge, x = 'b*', y = 'B_pred', hue = 'time')
plt.title(model)

# In[]

plt.style.use('seaborn')

L_lin = linregress(merge['L_pred'], merge['L*'])
A_lin = linregress(merge['A_pred'], merge['a*'])
B_lin = linregress(merge['B_pred'], merge['b*'])

fig, ax = plt.subplots(2,2)
fig.tight_layout(h_pad = 2)

ax[0,0].scatter(merge['L_pred'], merge['L*'])
ax[0,0].plot(merge['L_pred'], L_lin.slope * merge['L_pred'] + L_lin.intercept, color = 'indianred', label = str(round(L_lin.rvalue **2, 3)))
ax[0,0].set_title('L values')
ax[0,0].legend()

ax[0,1].scatter(merge['A_pred'], merge['a*'])
ax[0,1].plot(merge['A_pred'], A_lin.slope * merge['A_pred'] + A_lin.intercept, color = 'indianred', label = str(round(A_lin.rvalue **2, 3)))
ax[0,1].set_title('A values')
ax[0,1].legend()

ax[1,0].scatter(merge['B_pred'], merge['b*'])
ax[1,0].plot(merge['B_pred'], B_lin.slope * merge['B_pred'] + B_lin.intercept, color = 'indianred', label = str(round(B_lin.rvalue **2, 3)))
ax[1,0].set_title('B values')
ax[1,0].legend()

# In[]

plt.style.use('seaborn')

L_lin = linregress(pie_merge['Ls_mean'], pie_merge['L*'])
A_lin = linregress(pie_merge['As_mean'], pie_merge['a*'])
B_lin = linregress(pie_merge['Bs_mean'], pie_merge['b*'])

fig, ax = plt.subplots(2,2)
fig.tight_layout(h_pad = 2)

ax[0,0].scatter(pie_merge['Ls_mean'], pie_merge['L*'])
ax[0,0].plot(pie_merge['Ls_mean'], L_lin.slope * pie_merge['Ls_mean'] + L_lin.intercept, color = 'indianred', label = str(round(L_lin.rvalue **2, 3)))
ax[0,0].set_title('L values')
ax[0,0].legend()

ax[0,1].scatter(pie_merge['As_mean'], pie_merge['a*'])
ax[0,1].plot(pie_merge['As_mean'], A_lin.slope * pie_merge['As_mean'] + A_lin.intercept, color = 'indianred', label = str(round(A_lin.rvalue **2, 3)))
ax[0,1].set_title('A values')
ax[0,1].legend()

ax[1,0].scatter(pie_merge['Bs_mean'], pie_merge['b*'])
ax[1,0].plot(pie_merge['Bs_mean'], B_lin.slope * pie_merge['Bs_mean'] + B_lin.intercept, color = 'indianred', label = str(round(B_lin.rvalue **2, 3)))
ax[1,0].set_title('B values')
ax[1,0].legend()

# In[]

pie_merge.columns
sns.boxplot(pie_merge['b*'])
sns.boxplot(pie_merge['a*'])
sns.boxplot(pie_merge['L*'])

sns.swarmplot(pie_merge['b*'], color = 'red')
sns.boxplot(pie_merge['b*'])


merge.columns
short = merge.loc[merge['bake_time'] == 15]
sns.scatterplot(data = short, x = 'b*', y = 'B_pred', hue = 'ph_x', style = 'pea')
sns.scatterplot(data = short, x = 'a*', y = 'A_pred', hue = 'ph_x', style = 'pea')
sns.scatterplot(data = short, x = 'b*', y = 'B_pred', hue = 'ph_x', style = 'pea')


























