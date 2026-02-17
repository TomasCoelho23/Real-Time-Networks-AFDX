################################################################@
"""

This file is a starting base for xml network parsing.
It is provided in the scope of the AFDX Project (Open Timaeus-Net#414 Extension)
The aim of such a file it to simplify the python coding, so that students focus on Network Calculus topics.

You have to update and complete this file in order to fit all the projects requirements.
Particularly, you need to complete the Station, Switch, Edge, Flow and Target classes.

"""
################################################################@

import xml.etree.ElementTree as ET
import os.path
import sys
import random

################################################################@
""" Local classes """
################################################################@

""" Node
    The Node class is used to handle any node if the network
    It's an abstract class
"""
class Node:
    def __init__(self, name):
        self.name = name

""" Station
    The Station class is used to handle stations
"""
class Station(Node):
    def __init__(self, name):
        self.name = name
    def isSwitch(self):
        return False

""" Switch
    The Switch class is used to handle switches
"""
class Switch(Node):
    def __init__(self, name, latency):
        self.name = name
        self.latency = latency
    def isSwitch(self):
        return True

""" Edge
    The Edge class is used to handle edges
"""
class Edge:
    def __init__(self, name, frm,to):
        self.name = name
        self.frm = frm
        self.to = to

""" Target
    The Target class is used to handle targets
"""
class Target:
    def __init__(self, flow, to):
        self.flow = flow
        self.to = to
        self.path = []

""" Flow
    The Flow class is used to handle flows
"""
class Flow:
    targets = []
    def __init__(self, name, source, payload, overhead, period):
        self.name = name
        self.source = source
        self.payload = payload
        self.overhead = overhead
        self.period = period

################################################################@
""" Local methods """
################################################################@
""" parseStations
    Method to parse stations
        root : the xml main root
"""
def parseStations(root):
    for station in root.findall('station'):
        nodes.append (Station(station.get('name')))

""" parseSwitches
    Method to parse switches
        root : the xml main root
"""
def parseSwitches(root):
    for sw in root.findall('switch'):
        nodes.append (Switch(sw.get('name'),float(sw.get('tech-latency'))*1e-6))

""" parseEdges
    Method to parse edges
        root : the xml main root
"""
def parseEdges(root):
    for sw in root.findall('link'):
        edges.append (Edge(sw.get('name'),sw.get('from'),sw.get('to')))

""" parseFlows
    Method to parse flows
        root : the xml main root
"""
def parseFlows(root):
    for sw in root.findall('flow'):
        flow = Flow (sw.get('name'),sw.get('source'),float(sw.get('max-payload')),67,float(sw.get('period'))*1e-3)
        flows.append (flow)
        for tg in sw.findall('target'):
            target = Target(flow,tg.get('name'))
            flow.targets.append(target)
            target.path.append(flow.source)
            for pt in tg.findall('path'):
                target.path.append(pt.get('node'))

""" parseNetwork
    Method to parse the whole network
        xmlFile : the path to the xml file
"""
def parseNetwork(xmlFile):
    if os.path.isfile(xmlFile):
        tree = ET.parse(xmlFile)
        root = tree.getroot()
        parseStations(root)
        parseSwitches(root)
        parseEdges(root)
        parseFlows(root)
    else:
        print("File not found: "+xmlFile)

""" traceNetwork
    Method to trace the network to the console
"""
def traceNetwork():
    print("Stations:")
    for node in nodes:
        if not node.isSwitch():
            print ("\t" + node.name)
            
    print("\nSwitches:")
    for node in nodes:
        if node.isSwitch():
            print ("\t" + node.name)
            
    print("\nEdges:")
    for edge in edges:
        print ("\t" + edge.name + ": " + edge.frm + "=>" + edge.to)
    
    print("\nFlows:")
    for flow in flows:
        print ("\t" + flow.name + ": " + flow.source + " (L=" + str(flow.payload) +", p=" + str(flow.period) + ")")
        for target in flow.targets:
            print ("\t\tTarget=" + target.to)
            for node in target.path:
                print ("\t\t\t" + node)

""" createFakeResultsFile
    Method to create a fake result file ; only delays are generated (random value between 40 and 80)
        xmlFile : the path to the xml (input) file
"""
def createFakeResultsFile (xmlFile):
    posDot = xmlFile.rfind('.')
    if not (posDot == -1):
        resFile = xmlFile[0:posDot]+'_res.xml'
    else:
        resFile = xmlFile+'_res.xml'
    res = open(resFile,"w")
    res.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    res.write('<results>\n')
    res.write('\t<delays>\n')
    for flow in flows:
        res.write('\t\t<flow name="' + flow.name + '">\n');
        for target in flow.targets:
            res.write('\t\t\t<target name="' + target.to + '" value="'+str(random.randint(400, 800))+'" />\n');
        res.write('\t\t</flow>\n')
    res.write('\t</delays>\n')
    res.write('</results>\n')
    res.close()
    file2output(resFile)
    
""" file2output
    Method to print a file to standard ouput
        file : the path to the xml (input) file
"""
def file2output (file):
    hFile = open(file, "r")
    for line in hFile:
        print(line.rstrip())

################################################################@
""" Global data """
################################################################@
nodes = [] # the nodes
edges = [] # the edges
flows = [] # the flows

################################################################@
""" Main program """
################################################################@

if len(sys.argv)>=2:
    xmlFile=sys.argv[1]
else:
    xmlFile="./ES2E_M.xml"
    
parseNetwork(xmlFile)
#traceNetwork()
createFakeResultsFile(xmlFile)


