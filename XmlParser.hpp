#pragma once

#include "tinyxml2.h"

typedef tinyxml2::XMLVisitor XmlVis;
typedef tinyxml2::XMLNode XmlNode;
typedef tinyxml2::XMLText XmlTxt;

typedef tinyxml2::XMLComment XmlCom;
typedef tinyxml2::XMLDeclaration XmlDec;
typedef tinyxml2::XMLUnknown XmlUnk;

typedef tinyxml2::XMLAttribute XmlAttr;
typedef tinyxml2::XMLHandle XmlHand;
typedef tinyxml2::XMLConstHandle XmlConstHand;

typedef tinyxml2::XMLDocument XmlDoc;
typedef tinyxml2::XMLElement XmlElem;
typedef tinyxml2::XMLPrinter XmlPrinter;

typedef tinyxml2::XMLError XmlError;
const XmlError XmlSuccess = tinyxml2::XML_SUCCESS;
