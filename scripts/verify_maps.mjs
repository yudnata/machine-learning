import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const reports=[];

function gridHarness(html) {
  const start=html.indexOf('const gridMap = ');
  const finish=html.indexOf('gridToggleControl.addTo(gridMap);',start)+'gridToggleControl.addTo(gridMap);'.length;
  assert.ok(start>=0 && finish>start,'Grid toggle script missing');
  const code=html.slice(start,finish);
  const mapName=code.match(/const gridMap = (map_\w+)/)[1];
  const pairs=[...code.matchAll(/"([^"]+)": (feature_group_\w+|geo_json_\w+)/g)];
  const initialKey=JSON.parse(code.match(/currentLayer:\s*gridLayers\[("[^"]+")\]/)[1]);
  const nodes=[];
  const element=tag=>{
    const e={tag,children:[],style:{},attributes:{},appendChild(child){this.children.push(child);},setAttribute(key,value){this.attributes[key]=value;}};
    nodes.push(e);return e;
  };
  const land={name:'land_context'};
  const active=new Set([land]);
  const map={hasLayer:layer=>active.has(layer),removeLayer:layer=>active.delete(layer),addLayer:layer=>active.add(layer),closePopup(){this.popupClosed=true;}};
  const context={document:{createElement:element,createTextNode:text=>({text})},L:{
    control:()=>({addTo(){this.onAdd();}}),DomUtil:{create:element},
    DomEvent:{disableClickPropagation(){},disableScrollPropagation(){}}
  }};
  context[mapName]=map;
  for(const [_,key,name] of pairs)context[name]={key};
  active.add(context[pairs.find(p=>p[1]===initialKey)[2]]);
  vm.runInNewContext(code,context);
  const button=nodes.find(e=>e.tag==='button');
  assert.ok(button);
  assert.equal(button.type,'button');
  assert.equal(button.attributes['aria-pressed'],'true');
  assert.equal(button.textContent,'Sembunyikan grid');
  return {nodes,active,land,map,context,button};
}

for (const method of ['kmeans','dbscan']) {
  const html=fs.readFileSync(path.join(root,'maps',`${method}_profiles_by_quarter.html`),'utf8');
  for (const file of [`${method}_profiles_by_quarter.html`,`${method}_profile_changes_2021_2025.html`]) {
    const content=fs.readFileSync(path.join(root,'maps',file),'utf8');
    assert.ok(!/<script\b[^>]*\bsrc\s*=/i.test(content),`${file}: remote script`);
    assert.ok(!/<link\b[^>]*\brel=["']stylesheet/i.test(content),`${file}: remote CSS`);
    assert.ok(!content.includes('tile.openstreetmap.org'),`${file}: OSM tiles`);
    assert.ok(!content.includes('L.tileLayer('),`${file}: tile layer`);
    assert.ok(content.includes('land_context'),`${file}: local land missing`);
    assert.ok(content.includes('.fitBounds('),`${file}: initial bounds missing`);
  }
  const begin=html.lastIndexOf('const mapRef = ');
  const finish=html.indexOf('selector.addTo(mapRef);',begin)+'selector.addTo(mapRef);'.length;
  assert.ok(begin>=0 && finish>begin);
  const code=html.slice(begin,finish);
  const mapName=code.match(/const mapRef = (map_\w+)/)[1];
  const pairs=[...code.matchAll(/"(\d{4}Q[1-4])": (feature_group_\w+)/g)];
  assert.equal(pairs.length,24);
  const {nodes,active,land,map,context,button}=gridHarness(html);
  assert.equal(context[mapName],map);
  const defaultPeriod=code.match(/opt.selected=p==="(\d{4}Q[1-4])"/)[1];
  assert.ok(active.has(context[pairs.find(p=>p[1]===defaultPeriod)[2]]));
  vm.runInNewContext(code,context);
  const select=nodes.find(e=>e.tag==='select');
  assert.equal(select.attributes['aria-label'],'Tahun dan triwulan');
  assert.equal(select.children.length,24);
  assert.equal(select.children.find(o=>o.value==='2026Q3').textContent,'Juli-September 2026');
  assert.ok(select.children.find(o=>o.value==='2026Q4').textContent.includes('belum lengkap'));
  assert.ok(nodes.some(e=>e.textContent?.includes('Algoritma membentuk kelompok')));
  assert.ok(!nodes.some(e=>e.textContent?.includes('Cara menjelaskan kepada dosen')));
  for(const [_,period,name] of pairs){
    select.value=period;
    select.onchange();
    assert.equal(active.size,2);
    assert.ok(active.has(land));
    assert.ok(active.has(context[name]));
  }
  button.onclick();
  assert.equal(button.textContent,'Tampilkan grid');
  assert.equal(button.attributes['aria-pressed'],'false');
  assert.equal(active.size,1);
  assert.ok(active.has(land));
  assert.equal(map.popupClosed,true);
  for(const [_,period] of pairs){
    select.value=period;select.onchange();
    assert.equal(active.size,1,'Changing period must not reveal a hidden grid');
    assert.ok(active.has(land));
  }
  button.onclick();
  assert.equal(button.textContent,'Sembunyikan grid');
  assert.equal(button.attributes['aria-pressed'],'true');
  assert.equal(active.size,2);
  assert.ok(active.has(context[pairs.at(-1)[2]]),'Show grid must restore the latest selected period');
  const changeHtml=fs.readFileSync(path.join(root,'maps',`${method}_profile_changes_2021_2025.html`),'utf8');
  const change=gridHarness(changeHtml);
  change.button.onclick();
  assert.equal(change.active.size,1);
  assert.ok(change.active.has(change.land));
  change.button.onclick();
  assert.equal(change.active.size,2);
  assert.ok(change.active.has(change.land));
  reports.push({method,status:'PASS',period_count:24,selector_one_active_layer:'PASS',grid_toggle:'PASS',period_change_while_hidden:'PASS',changes_map_grid_toggle:'PASS',basemap_stays_visible:'PASS',aria_label:'PASS',offline_assets:'PASS',tile_requests:0});
}
fs.writeFileSync(path.join(root,'reports','validation','map_selector.json'),JSON.stringify({
  status:'PASS',scope:'JavaScript kontrol peta diuji dengan DOM/Leaflet stub, bukan browser nyata',
  browser_check:'Tidak tersedia: provider browser iab tidak aktif dalam sesi ini',maps:reports
},null,2));
console.log(JSON.stringify(reports,null,2));
