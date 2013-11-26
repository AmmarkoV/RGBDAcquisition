#include "SelectAcquisitionGraph.h"

//(*InternalHeaders(SelectAcquisitionGraph)
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(SelectAcquisitionGraph)
const long SelectAcquisitionGraph::ID_COMBOBOX1 = wxNewId();
const long SelectAcquisitionGraph::ID_PANEL1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(SelectAcquisitionGraph,wxDialog)
	//(*EventTable(SelectAcquisitionGraph)
	//*)
END_EVENT_TABLE()

SelectAcquisitionGraph::SelectAcquisitionGraph(wxWindow* parent,wxWindowID id)
{
	//(*Initialize(SelectAcquisitionGraph)
	Create(parent, id, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("id"));
	SetClientSize(wxSize(933,627));
	Panel1 = new wxPanel(this, ID_PANEL1, wxPoint(40,264), wxSize(224,144), wxTAB_TRAVERSAL, _T("ID_PANEL1"));
	ComboBox1 = new wxComboBox(Panel1, ID_COMBOBOX1, wxEmptyString, wxPoint(16,24), wxSize(192,25), 0, 0, 0, wxDefaultValidator, _T("ID_COMBOBOX1"));
	ComboBox1->Append(_("NONE"));
	ComboBox1->Append(_("V4L2"));
	ComboBox1->Append(_("V4L2 STEREO"));
	ComboBox1->Append(_("FREENECT"));
	ComboBox1->Append(_("OPENNI1"));
	ComboBox1->Append(_("OPENNI2"));
	ComboBox1->Append(_("OPENGL"));
	ComboBox1->Append(_("TEMPLATE"));
	ComboBox1->Append(_("NETWORK"));
	//*)
}

SelectAcquisitionGraph::~SelectAcquisitionGraph()
{
	//(*Destroy(SelectAcquisitionGraph)
	//*)
}

