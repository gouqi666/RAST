import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
x = [0.05,0.1,0.15,0.20,0.25]
EM = [69.19,63.17,63.57,61.29,60.03] #
F1 = [85.16,80.24,81.08,79.11,78.2]
P_BLEU = [74.7,48.91,46.6,38.17,35.18]
TOP1_BLEU = [20.78,19.25,15.51,14.35,13.39]
ORACLE_BLEU = [23.97,23.23,23.08,22.73,22.63]
for i in range(5):
    EM[i] = EM[i] / 100
    F1[i] = F1[i] / 100
    P_BLEU[i] = P_BLEU[i] / 100
    TOP1_BLEU[i] = TOP1_BLEU[i] / 100
    ORACLE_BLEU[i] = ORACLE_BLEU[i] / 100
plt.figure()
plt.xlabel("Different diverse coefficient")
plt.ylabel("Score")
plt.plot(x,EM,marker='*',label='EM')
plt.plot(x,F1,marker='o',label='F1')
plt.plot(x,P_BLEU,marker='^',label='P_BLEU')
plt.plot(x,TOP1_BLEU,marker='H',label='TOP_1_BLEU')
plt.plot(x,ORACLE_BLEU,marker='P',label='ORACLE_BLEU')
gap = 0.005


plt.annotate(xy=(x[0] - gap, EM[0]-0.035), text='{:.2}'.format(EM[0]))
plt.annotate(xy=(x[4] - gap, EM[4]+0.015), text='{:.2}'.format(EM[4]))

plt.annotate(xy=(x[0]-gap, F1[0]-0.035), text='{:.2}'.format(F1[0]))
plt.annotate(xy=(x[4]-gap, F1[4]+0.012), text='{:.2}'.format(F1[4]))

plt.annotate(xy=(x[0]-gap, P_BLEU[0]+0.015), text='{:.2}'.format(P_BLEU[0]))
plt.annotate(xy=(x[4]-gap, P_BLEU[4]+0.015), text='{:.2}'.format(P_BLEU[4]))

plt.annotate(xy=(x[0]-0.005, TOP1_BLEU[0]-0.05), text='{:.2}'.format(TOP1_BLEU[0]))
plt.annotate(xy=(x[4]-gap, TOP1_BLEU[4]+0.02), text='{:.2}'.format(TOP1_BLEU[4]))

plt.annotate(xy=(x[0]-gap, ORACLE_BLEU[0]+0.015), text='{:.2}'.format(ORACLE_BLEU[0]))
plt.annotate(xy=(x[4]-gap, ORACLE_BLEU[4]+0.015), text='{:.2}'.format(ORACLE_BLEU[4]))

plt.legend(bbox_to_anchor=(0.26,0.5),prop = {'size':8})
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.savefig('./diverse_coef')
plt.show()

