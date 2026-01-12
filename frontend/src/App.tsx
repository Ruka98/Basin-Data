
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { Layout, Cloud, Droplets, Sun, Map, FileText } from 'lucide-react';
import { cn } from './lib/utils';

// --- Types ---

interface YearRange {
  start_year: number;
  end_year: number;
  available_years: number[];
  files_found: {
    P: boolean;
    ET: boolean;
    LU: boolean;
  };
}

interface OverviewMetrics {
  metrics: { [key: string]: number };
  intro: string;
  summary_items: string[];
}

interface ChartData {
  map_figure: any;
  bar_figure: any;
  explanation: string;
}

const API_BASE = '/api';

// --- Components ---

const Card = ({ children, className }: { children: React.ReactNode; className?: string }) => (
  <div className={cn("bg-white rounded-xl shadow-sm border border-slate-200 p-6", className)}>
    {children}
  </div>
);

const SectionTitle = ({ children, icon: Icon }: { children: React.ReactNode; icon: any }) => (
  <h3 className="text-xl font-semibold text-slate-800 mb-4 flex items-center gap-2 border-l-4 border-blue-500 pl-3">
    <Icon className="w-6 h-6 text-slate-500" />
    {children}
  </h3>
);

const MetricCard = ({ title, value, unit, icon: Icon, color, percentage }: any) => {
    if (value === undefined || value === 0) return null;

    let formattedValue = "0";
    if (Math.abs(value) < 0.001) formattedValue = "0";
    else if (Math.abs(value) < 1) formattedValue = value.toFixed(3);
    else if (Math.abs(value) < 10) formattedValue = value.toFixed(2);
    else if (Math.abs(value) < 100) formattedValue = value.toFixed(1);
    else formattedValue = value.toFixed(0);

    return (
        <div className="bg-white border-2 rounded-xl p-5 flex flex-col justify-between min-w-[200px] flex-1 min-h-[120px]" style={{ borderColor: `${color}20` }}>
            <div className="flex items-center gap-3">
                <span className="text-2xl">{Icon}</span>
                <div>
                    <h4 className="m-0 text-base font-semibold text-slate-700">{title}</h4>
                    <p className="m-0 text-2xl font-bold mt-1" style={{ color: color }}>
                        {formattedValue} <span className="text-sm font-normal text-slate-500">{unit}</span>
                    </p>
                </div>
            </div>
            {percentage !== undefined && (
                <p className="text-xs text-slate-500 mt-2">{percentage.toFixed(1)}% of inflows</p>
            )}
        </div>
    )
}

function App() {
  const [basins, setBasins] = useState<string[]>([]);
  const [selectedBasin, setSelectedBasin] = useState<string | null>(null);
  const [yearRange, setYearRange] = useState<YearRange | null>(null);
  const [selectedStartYear, setSelectedStartYear] = useState<number | null>(null);
  const [selectedEndYear, setSelectedEndYear] = useState<number | null>(null);
  const [overview, setOverview] = useState<OverviewMetrics | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'landuse' | 'precipitation' | 'evapotranspiration' | 'balance'>('overview');

  const [pData, setPData] = useState<ChartData | null>(null);
  const [etData, setEtData] = useState<ChartData | null>(null);
  const [petData, setPetData] = useState<ChartData | null>(null);
  const [luData, setLuData] = useState<ChartData | null>(null);

  // --- Initial Data Load ---
  useEffect(() => {
    axios.get(`${API_BASE}/basins`).then(res => {
      setBasins(res.data);
      if (res.data.length > 0) setSelectedBasin(res.data[0]);
    });
    // Removed unused map fetch
  }, []);

  // --- Basin Selection Change ---
  useEffect(() => {
    if (!selectedBasin) return;

    // Fetch Years
    axios.get(`${API_BASE}/basins/${selectedBasin}/years`).then(res => {
      setYearRange(res.data);
      const available = res.data.available_years;
      if (available.length > 0) {
        // Default to last 3 years or all if less
        const end = available[available.length - 1];
        const start = available.length > 2 ? available[available.length - 3] : available[0];
        setSelectedStartYear(start);
        setSelectedEndYear(end);
      }
    }).catch(err => {
        console.error(err);
    });
  }, [selectedBasin]);

  // --- Fetch Data Based on Selection ---
  useEffect(() => {
    if (!selectedBasin || !selectedStartYear || !selectedEndYear) return;

    const fetchData = async () => {
        try {
            // Overview
            const overviewRes = await axios.get(`${API_BASE}/basins/${selectedBasin}/overview`, {
                params: { start_year: selectedStartYear, end_year: selectedEndYear }
            });
            setOverview(overviewRes.data);
        } catch (e) {
            console.error(e);
        }
    };

    fetchData();
  }, [selectedBasin, selectedStartYear, selectedEndYear]);

  // Fetch chart data when tab changes or selection changes
  useEffect(() => {
      if (!selectedBasin || !selectedStartYear || !selectedEndYear) return;

      const fetchChart = async (_type: string, setter: any, endpoint: string) => {
          try {
              const res = await axios.get(endpoint, {
                  params: { start_year: selectedStartYear, end_year: selectedEndYear }
              });
              setter(res.data);
          } catch(e) { console.error(e); }
      };

      if (activeTab === 'precipitation' && !pData) fetchChart('P', setPData, `${API_BASE}/basins/${selectedBasin}/hydro/P`);
      if (activeTab === 'evapotranspiration' && !etData) fetchChart('ET', setEtData, `${API_BASE}/basins/${selectedBasin}/hydro/ET`);
      if (activeTab === 'balance' && !petData) fetchChart('P-ET', setPetData, `${API_BASE}/basins/${selectedBasin}/hydro/P-ET`);
      if (activeTab === 'landuse' && !luData) {
          axios.get(`${API_BASE}/basins/${selectedBasin}/landuse`).then(res => setLuData(res.data)).catch(console.error);
      }
      // Reload on params change
      if (activeTab === 'precipitation') fetchChart('P', setPData, `${API_BASE}/basins/${selectedBasin}/hydro/P`);
      if (activeTab === 'evapotranspiration') fetchChart('ET', setEtData, `${API_BASE}/basins/${selectedBasin}/hydro/ET`);
      if (activeTab === 'balance') fetchChart('P-ET', setPetData, `${API_BASE}/basins/${selectedBasin}/hydro/P-ET`);
      if (activeTab === 'landuse') axios.get(`${API_BASE}/basins/${selectedBasin}/landuse`).then(res => setLuData(res.data)).catch(console.error);

  }, [activeTab, selectedBasin, selectedStartYear, selectedEndYear]);


  const ChartSection = ({ title, data, icon }: { title: string, data: ChartData | null, icon: any }) => {
      if (!data) return <div className="p-10 text-center text-slate-500">Loading data...</div>;

      return (
          <Card className="mt-6">
              <SectionTitle icon={icon}>{title}</SectionTitle>
              <div className="flex flex-col lg:flex-row gap-6">
                  <div className="flex-1 min-h-[400px]">
                      <Plot
                          data={data.map_figure.data}
                          layout={{ ...data.map_figure.layout, autosize: true }}
                          useResizeHandler={true}
                          style={{ width: "100%", height: "100%" }}
                          config={{ responsive: true }}
                      />
                  </div>
                  <div className="flex-1 min-h-[400px]">
                      <Plot
                          data={data.bar_figure.data}
                          layout={{ ...data.bar_figure.layout, autosize: true }}
                          useResizeHandler={true}
                          style={{ width: "100%", height: "100%" }}
                          config={{ responsive: true }}
                      />
                  </div>
              </div>
              <div className="mt-4 p-4 bg-slate-50 rounded-lg text-slate-600 text-sm leading-relaxed border border-slate-100">
                  <div dangerouslySetInnerHTML={{ __html: data.explanation.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
              </div>
          </Card>
      );
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 pb-20">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div className="flex items-center gap-3">
                <div className="bg-blue-600 text-white p-2 rounded-lg">
                    <Droplets className="w-6 h-6" />
                </div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                    Basin Data Dashboard
                </h1>
            </div>

            <div className="flex items-center gap-4">
                <select
                    className="form-select block w-full pl-3 pr-10 py-2 text-base border-slate-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md border"
                    value={selectedBasin || ''}
                    onChange={(e) => setSelectedBasin(e.target.value)}
                >
                    {basins.map(b => <option key={b} value={b}>{b}</option>)}
                </select>

                {yearRange && (
                    <div className="flex items-center gap-2 text-sm bg-slate-100 p-1 rounded-md border border-slate-200">
                        <select
                            className="bg-transparent border-none focus:ring-0 py-1 pl-2 pr-6 text-sm"
                            value={selectedStartYear || ''}
                            onChange={(e) => setSelectedStartYear(Number(e.target.value))}
                        >
                             {yearRange.available_years.map(y => <option key={y} value={y}>{y}</option>)}
                        </select>
                        <span className="text-slate-400">to</span>
                        <select
                            className="bg-transparent border-none focus:ring-0 py-1 pl-2 pr-6 text-sm"
                            value={selectedEndYear || ''}
                            onChange={(e) => setSelectedEndYear(Number(e.target.value))}
                        >
                            {yearRange.available_years.map(y => <option key={y} value={y}>{y}</option>)}
                        </select>
                    </div>
                )}
            </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

        {/* Navigation Tabs */}
        <div className="flex space-x-1 rounded-xl bg-slate-200 p-1 mb-8 overflow-x-auto">
            {[
                { id: 'overview', label: 'Overview', icon: Layout },
                { id: 'landuse', label: 'Land Use', icon: Map },
                { id: 'precipitation', label: 'Precipitation', icon: Cloud },
                { id: 'evapotranspiration', label: 'Evapotranspiration', icon: Sun },
                { id: 'balance', label: 'Water Balance', icon: Droplets },
            ].map((tab) => (
                <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={cn(
                        "flex items-center gap-2 w-full rounded-lg py-2.5 text-sm font-medium leading-5 ring-white ring-opacity-60 ring-offset-2 ring-offset-blue-400 focus:outline-none focus:ring-2",
                        activeTab === tab.id
                            ? "bg-white text-blue-700 shadow"
                            : "text-slate-600 hover:bg-white/[0.12] hover:text-slate-800"
                    )}
                >
                    <tab.icon className="w-4 h-4 ml-3" />
                    <span className="mr-3 whitespace-nowrap">{tab.label}</span>
                </button>
            ))}
        </div>

        {/* Content Area */}
        <div className="space-y-6">

            {activeTab === 'overview' && overview && (
                <div className="animate-in fade-in duration-500 slide-in-from-bottom-4">
                    {/* Intro */}
                    {overview.intro && (
                        <Card className="mb-6">
                            <SectionTitle icon={FileText}>Introduction</SectionTitle>
                            <p className="text-slate-600 leading-relaxed whitespace-pre-line">{overview.intro}</p>
                        </Card>
                    )}

                    {/* Summary List */}
                    <Card className="mb-6 bg-slate-50 border-l-4 border-l-blue-500">
                        <h4 className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
                             Executive Summary
                        </h4>
                        <ul className="space-y-2">
                            {overview.summary_items.map((item, idx) => (
                                <li key={idx} className="flex gap-2 text-slate-700 text-sm">
                                    <span className="text-blue-500 mt-1">•</span>
                                    {item}
                                </li>
                            ))}
                        </ul>
                    </Card>

                    {/* Metrics Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        <MetricCard title="Total Inflows" value={overview.metrics.total_inflows} unit="Mm³/yr" icon="🌊" color="#3b82f6" />
                        <MetricCard title="Precipitation" value={overview.metrics.total_precipitation} unit="Mm³/yr" icon="🌧️" color="#06b6d4" percentage={overview.metrics.precipitation_percentage} />
                        <MetricCard title="Water Imports" value={overview.metrics.surface_water_imports} unit="Mm³/yr" icon="🚚" color="#8b5cf6" />
                        <MetricCard title="Total Consumption" value={overview.metrics.total_water_consumption} unit="Mm³/yr" icon="💧" color="#ef4444" />
                        <MetricCard title="Manmade Cons." value={overview.metrics.manmade_consumption} unit="Mm³/yr" icon="🏭" color="#f59e0b" />
                        <MetricCard title="Treated Wastewater" value={overview.metrics.treated_wastewater} unit="Mm³/yr" icon="♻️" color="#10b981" />
                        <MetricCard title="Non-irrigated Cons." value={overview.metrics.non_irrigated_consumption} unit="Mm³/yr" icon="🏘️" color="#6366f1" />
                        <MetricCard title="Recharge" value={overview.metrics.recharge} unit="Mm³/yr" icon="⤵️" color="#06b6d4" />
                    </div>
                </div>
            )}

            {activeTab === 'landuse' && (
                <div className="animate-in fade-in duration-500">
                     <ChartSection title="Land Use / Land Cover" data={luData} icon={Map} />
                </div>
            )}

            {activeTab === 'precipitation' && (
                <div className="animate-in fade-in duration-500">
                     <ChartSection title="Precipitation" data={pData} icon={Cloud} />
                </div>
            )}

            {activeTab === 'evapotranspiration' && (
                <div className="animate-in fade-in duration-500">
                     <ChartSection title="Evapotranspiration" data={etData} icon={Sun} />
                </div>
            )}

            {activeTab === 'balance' && (
                <div className="animate-in fade-in duration-500">
                     <ChartSection title="Water Balance (P - ET)" data={petData} icon={Droplets} />
                </div>
            )}

        </div>
      </main>
    </div>
  );
}

export default App;
