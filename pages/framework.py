from dash import html, dcc

WA_FRAMEWORK_TEXT = """
**∆S/∆t = P - ET - Q_out**                                                                                   (1)

Where:
*   **∆S** is the change in storage
*   **∆t** is the change in time
*   **P** is precipitation (mm/year or m3/year)
*   **ET** is total actual evapotranspiration (mm/year or m3/year)
*   **Qout** is total surface water outflow (mm/year or m3/year)

To utilize the WA+ approach for water budget reporting in Jordan, it is important to account for all water users, other than irrigation, and their return flows into equation 1. Also, in Jordan, man-made inflows and outflows of great importance especially in heavily populated basins (Amdar et al., 2024). Therefore, an updated water balance incorporating various sectoral water consumption in addition to inflow and outflows is proposed (Amdar et al., 2024). Hence, equation (2) represents the updated WA+ water balance equation in the context of Jordan. This modification will further be refined following detailed discussions and consultations with the WEC and MWI team to ensure complete understanding and consensus of the customized framework for Jordan.

**∆S/∆t = (P + Q_in) - (ET + CW_sec + Q_WWT + Q_re + Q_natural)**                               (2)

where:
*   **P** is the total precipitation (Mm3/year)
*   **ET** is the total actual evapotranspiration (Mm3/year)
*   **Qin** is the total inflows into the basin consisting of both surface water inflows and any other inter-basin transfers (Mm3/year)
*   **Qre** is the total recharge to groundwater from precipitation and return flow (Mm3/year)
*   **QWWT** is the total treated waste water that is returned to the river system after treatment. This could be from domestic, industry and tourism sectors (Mm3/year)
*   **Qnatural** is the naturalized streamflow from the basin (Mm3/year)
*   **CWsec** is the total non-irrigated water use/consumption (ie water that is not returned to the system but is consumed by humans) and is given by:

**CWsec = Supplydomestic + Supplyindustrial + Supplylivestock + Supplytourism**
(3)

Where:
*   **Supplydomestic** is the water supply for the domestic sector (Mm3/year)
*   **Supplyindustrial** is the water supply for the industrial sector (Mm3/year)
*   **Supplylivestock** is the water supply for the livestock sector (Mm3/year)
*   **Supplytourism** is the water supply for the tourism sector (Mm3/year)

The customized WA+ framework thus takes into account both agricultural and non-irrigated water consumption, water imports and the return of treated wastewater into the basin.
"""

layout = html.Div(className="dashboard-container", children=[
    html.Div(className="graph-card", style={"marginTop": "20px"}, children=[
        html.H2("Customized WA+ Analytics for Jordan", style={"color": "#315F83", "marginBottom": "20px"}),
        dcc.Markdown(WA_FRAMEWORK_TEXT, className="markdown-content")
    ])
])
