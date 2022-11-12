let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

function getMousePos(e) {
  const rect = canvas.getBoundingClientRect();
  return {
      x: (e.clientX - rect.left) / (rect.right - rect.left) * canvas.width,
      y: (e.clientY - rect.top) / (rect.bottom - rect.top) * canvas.height
  };
}

class Canvas {
  constructor(){
    this.mouse = new Vector();
    this.isMouseDown = false;

    window.addEventListener('resize', this.resizeCanvas, false);
    this.resizeCanvas();
    
    canvas.addEventListener('mousemove', (e) => { this.mouseMove(e); }, false);
    canvas.addEventListener('mousedown', () => { this.mouseDown(); }, false);
    canvas.addEventListener('mouseup', () => { this.mouseUp(); }, false);
  }

  resizeCanvas() {
    const parentElement = canvas.parentNode;
    const parentWidth = parentElement.clientWidth;
    const parentStyle = getComputedStyle(parentElement);
    const parentPaddingLeft = parseFloat(parentStyle.getPropertyValue('padding-left'));
    const parentPaddingRight = parseFloat(parentStyle.getPropertyValue('padding-right'));
    const viewportOffset = canvas.getBoundingClientRect();
    const offsetTop = viewportOffset.top;
    const windowHeight = window.innerHeight;

    canvas.width = parentWidth - parentPaddingLeft - parentPaddingRight;

    if(canvas.width > windowHeight - offsetTop){
      canvas.height =  windowHeight - offsetTop - 60;
    } else {
      canvas.height = canvas.width;
    }
  }

  mouseMove(e){
    const pos = getMousePos(e);
    this.mouse.setXY(pos.x, -pos.y)
  }

  mouseDown(){
    this.isMouseDown = true;
  }

  mouseUp(){
    this.isMouseDown = false;
  }

  getMouse(){
    return {
      pos: this.mouse,
      isMouseDown: this.isMouseDown
    }
  }
}

class Grid {
  constructor(){
    this.offset = new Vector(100, 100);
    this.gridDim = new Vector(300, 300);
    this.origin = new Vector(200, 200);
    this.zoomSteps = 10;
    this.zoomStep = 5;
    this.scaleStep = 0;
    this.scale = new Vector(1, 1);
    this.ppm = 62.5; // px per meter
    this.spacing1 = 100;
    this.color1 = "#999999";
    this.color2 = "#dddddd";
    this.color3 = "#000000";
    this.textColor = "#000000";
    this.axisTextOffset = 20;

    this.mouse = null;
    this.lastDrag = null;

    this.velocity = new Vector();
    this.friction = 0.05;

    this.points = [];

    this.uiElement = document.getElementById("canvas-ui");

    canvas.addEventListener('wheel', (e) => {
      const dir = e.deltaY < 0 ? 1 : -1;
      this.zoomGrid(dir, true);
    }, { passive: true });
    this.uiElement.addEventListener('click', (e) => { this.onClick(e) }, { passive: true });
    this.applyZoom();
  }

  onClick(e){
    const target = e.target;
    const id = target.getAttribute("id");

    if(id === "button-home"){
      this.resetZoom();
    }

    if(id === "button-zoomin"){
      this.zoomGrid(-1);
    }

    if(id === "button-zoomout"){
      this.zoomGrid(1);
    }
  }

  applyZoom(deltaScale, withMouse){
    const zoomFactor = 1 + this.zoomStep / this.zoomSteps;
    const prevPpm = this.ppm;
    this.ppm = this.spacing1 * zoomFactor;

    if(this.mouse !== null){
      const mousePx = withMouse ? // Zoom to:
        Vector.sub(this.mouse, this.origin) : // Center off mouse
        Vector.sub(Vector.div(this.gridDim, 2), this.origin); // Center off canvas

      if(deltaScale.x === 0){
        const factor = 1 + (this.ppm - prevPpm) / prevPpm;
        const deltaMousePx = Vector.sub(mousePx, Vector.mult(mousePx, factor))

        this.origin.add(deltaMousePx);
      } else {
        const oldValue = Vector.div(Vector.div(mousePx, prevPpm), Vector.sub(this.scale, deltaScale));
        const newValue = Vector.div(Vector.div(mousePx, this.ppm), this.scale);
        const deltaPx = Vector.mult(Vector.mult(Vector.sub(newValue, oldValue), this.ppm), this.scale);
        this.origin.add(deltaPx);
      } 
    }
  }

  zoomGrid(dir, withMouse = false) {

    // First zoom a number of steps, then scale

    if(this.scaleStep === 41 && dir > 0){
      return;
    }

    this.zoomStep += dir;

    if(this.zoomStep > this.zoomSteps){
      this.zoomStep = 0;
      this.scaleStep += dir;
    }

    if(this.zoomStep < 0){
      this.zoomStep = this.zoomSteps;
      this.scaleStep += dir;
    }
    
    // At each zoom step, scale with fixed factors, to ensure no infinite number after comma
    //
    // -11    -10    -9    -8    -7    -6   -5   -4   -3  -2  -1  0 1 2 3  4  5  6   7   8   9    10   11
    // 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100 200 500 1000 2000

    let order = 1;

    if(this.scaleStep / 3 % 1 === 0){
      order = Math.pow(10, this.scaleStep / 3);
    }
    if((this.scaleStep - 1) / 3 % 1 === 0){
      order = 2 * Math.pow(10, (this.scaleStep - 1) / 3);
    }
    if((this.scaleStep - 2) / 3 % 1 === 0){
      order = 5 * Math.pow(10, (this.scaleStep - 2) / 3);
    }

    const deltaScale = Vector.sub(new Vector(order, order), this.scale);

    this.scale.setXY(order, order);

    this.applyZoom(deltaScale, withMouse);
  }

  resetZoom(){
    this.zoomStep = 5;
    this.scaleStep = 0;
    this.scale.setXY(1, 1);
    this.origin.setXY(0, 0);
    //this.ppm = 62.5; // px per meter
  }

  addPoint(x, y, vx, vy) {
    this.points.push(new Point(x, y, vx, vy));
  }
  updatePoints(dt) {
    for (let i = 0; i < this.points.length; i++) {
      this.points[i].update(dt);
    }
  }
  renderPoints() {
    for (let i = 0; i < this.points.length; i++) {
      this.points[i].render(this.origin, this.gridDim, this.ppm, this.scale);
    }
  }

  resize(width, height) {
    this.gridDim.setXY(width, height).sub(this.offset);
  }

  updateMouse(pos, active, dt) {
    this.mouse = Vector.add(pos, new Vector(-this.offset.x, this.gridDim.y));

    if(active){
      if(this.lastDrag !== null){
        const delta = Vector.sub(this.mouse, this.lastDrag);
        this.origin.add(delta);
        this.velocity = delta.div(dt);
      }
      this.lastDrag = this.mouse.copy();
    } else {
      this.lastDrag = null;
    }
  }

  updateOrigin(isMouseDown) {
    const velocityReduceConstant = 200;
    this.velocity.sub(Vector.mult(this.velocity, this.friction));
    if(!isMouseDown){
      this.origin.add(Vector.div(this.velocity, velocityReduceConstant))
    }
  }

  update(dt) {
    this.updatePoints(dt);
  }

  renderLine(start, end, color, lineWidth){
    ctx.beginPath();
    ctx.moveTo(start[0], start[1]);
    ctx.lineTo(end[0], end[1]);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
    ctx.closePath();
  }

  formatAxisNumber(num, x, y, horizontal = true){
    let formatNum = num;

    if(formatNum > 0.001 || formatNum < -0.001){
      formatNum = Math.round(num * 10000) / 10000;
    }

    let text1 = ctx.measureText(formatNum);
    let text2 = { width: 0 };

    // Scientific notation
    if(this.scaleX < 0.00001 || this.scaleX >= 10000 ||
      (formatNum <= 0.001 && formatNum >= -0.001 && formatNum !== 0)){
      let [coefficient, exponent] = 
      formatNum.toExponential().split('e').map(item => Number(item));

      const coeffString = coefficient.toString();

      if(coeffString.length > 5){
        coefficient = Math.round(coefficient * 10) / 10;
      }

      formatNum = `${coefficient} x 10`;
      
      text1 = ctx.measureText(formatNum);
      text2 = ctx.measureText(exponent);
      
      if(horizontal){
        ctx.fillText(exponent, x - text2.width, y - 6);
      } else {
        ctx.fillText(exponent, x + text1.width / 2, y - 1);
      }
    }

    if(horizontal){
      ctx.fillText(formatNum, x - text1.width - text2.width, y + 5);
    } else {
      ctx.fillText(formatNum, x, y + 10);
    }
  }

  render(){

    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = this.textColor;

    ctx.translate(this.offset.x, this.gridDim.y);

    const spacing = this.ppm;

    // Positive horizontal grid lines
    let y = this.origin.y;
    if(y > 0 && y < this.gridDim.y){
      ctx.fillText(0, -this.axisTextOffset, -y + 5);
      this.renderLine([0, -y], [this.gridDim.x, -y], this.color3, 1);
    }

    let i = this.origin.y < 0 ? 
    Math.floor(-this.origin.y / (spacing / 10)) : 1;

    while(y < this.gridDim.y) {
      y = this.origin.y + i * spacing / 10;
      if(y < 0) {
        i++;
        continue
      }
      if(i % 10 === 0){
        this.renderLine([0, -y], [this.gridDim.x, -y], this.color1, 1);
        const axisNumber = (y - this.origin.y) / spacing / this.scale.y;
        this.formatAxisNumber(axisNumber, -this.axisTextOffset, -y, true);
      } else {
        this.renderLine([0, -y], [this.gridDim.x, -y], this.color2, 1);
      }
      i++;
    }

    // Negative horizontal grid lines
    i = this.origin.y > this.gridDim.y ? 
    Math.floor((this.origin.y - this.gridDim.y) / (spacing / 10)) : 1;

    y = this.origin.y - i * spacing / 10;
    while(y > 0) {
      if(i % 10 === 0){
        this.renderLine([0, -y], [this.gridDim.x, -y], this.color1, 1);
        const axisNumber = (y - this.origin.y) / spacing / this.scale.y;
        this.formatAxisNumber(axisNumber, -this.axisTextOffset, -y, true);
      } else {
        this.renderLine([0, -y], [this.gridDim.x, -y], this.color2, 1);
      }
      i++;
      y = this.origin.y - i * spacing / 10;
    } 

    ctx.textAlign = 'center';

    // Positive vertical  grid lines
    let x = this.origin.x;
    if(x > 0 && x < this.gridDim.x){
      ctx.fillText(0, x, this.axisTextOffset);
      this.renderLine([x, 0], [x, -this.gridDim.y], this.color3, 1);
    }

    i = this.origin.x < 0 ? 
    Math.floor(-this.origin.x / (spacing / 10)) : 1;

    while(x < this.gridDim.x) {
      x = this.origin.x + i * spacing / 10;
      
      if(x < 0) {
        i++;
        continue;
      }
      if(i % 10 === 0){
        this.renderLine([x, 0], [x, -this.gridDim.y], this.color1, 1);
        const axisNumber = (x - this.origin.x) / spacing / this.scale.x;
        this.formatAxisNumber(axisNumber, x, this.axisTextOffset, false);
      } else {
        this.renderLine([x, 0], [x, -this.gridDim.y], this.color2, 1);
      }
      i++;
    }

    // Negative vertical  grid lines
    i = this.origin.x > this.gridDim.x ? 
    Math.floor(-(this.gridDim.x - this.origin.x) / (spacing / 10)) : 1;

    x = this.origin.x - i * spacing / 10;
    while(x > 0) {
      if(i % 10 === 0){
        this.renderLine([x, 0], [x, -this.gridDim.y], this.color1, 1);
        const axisNumber = (x - this.origin.x) / spacing / this.scale.x;
        this.formatAxisNumber(axisNumber, x, this.axisTextOffset, false);
      } else {
        this.renderLine([x, 0], [x, -this.gridDim.y], this.color2, 1);
      }
      
      i++;
      x = this.origin.x - i * spacing / 10;
    }

    this.renderPoints();

    ctx.translate(-this.offset.x, -this.gridDim.y);
  }
}

class Point {
  
  constructor(x, y, vx, vy) {
    this.pos = new Vector(x, y);
    this.vel = new Vector(vx, vy);

    this.gravity = new Vector(0, -9.81);

    this.path = [this.pos.copy()];
    this.pathColor = '#e62a4f';
    
    this.radius = 10;
    this.color = '#e62a4f';
    this.mass = 1;
  }
  
  update(dt) {
    this.vel.add(Vector.mult(this.gravity, dt));
    this.pos.add(Vector.mult(this.vel, dt));
    this.path.push(this.pos.copy());
  }

  getPrevPointOnGrid(prevPxPos, pxPos, gridDim){

    const rico = Vector.rico(prevPxPos, pxPos);

    if(pxPos.x < 0){
      pxPos.y += pxPos.x * -rico;
      pxPos.x = 0;
    }

    if(pxPos.x > gridDim.x){
      pxPos.y = prevPxPos.y + (prevPxPos.x - gridDim.x) * -rico;
      pxPos.x = gridDim.x;
    }

    if(pxPos.y < 0){
      pxPos.x = prevPxPos.x - prevPxPos.y / rico;
      pxPos.y = 0;
    }

    if(pxPos.y > gridDim.y){
      pxPos.x = prevPxPos.x + (gridDim.y - prevPxPos.y) / rico;
      pxPos.y = gridDim.y;
    }

    return pxPos;
  }

  isPointOnGrid(x, y, width, height){
    if(x >= 0 && x < width &&
      y <= 0 && y > -height){
      return true;
    }
    return false;
  }

  isPointOnGrid2(pos, dim){
    if(pos.x >= 0 && pos.x < dim.x &&
      pos.y >= 0 && pos.y < dim.y){
      return true;
    }
    return false;
  }

  mToPx(origin, pos, ppm, scale){
    return origin + pos * ppm * scale;
  }

  mToPx2(origin, pos, ppm, scale){
    return Vector.add(origin, Vector.mult(Vector.mult(pos, ppm), scale));
  }

  render(origin, gridDim, ppm, scale) {

    // Add circle to grid

    // Convert metric position to pixel position (meter to pixel)
    const pxPosCircle = this.mToPx2(origin, this.pos, ppm, scale);

    if(this.isPointOnGrid2(pxPosCircle, gridDim)){
      ctx.beginPath();
      ctx.fillStyle = this.color;
      ctx.arc(pxPosCircle.x, -pxPosCircle.y, this.radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.closePath();
    }

    // Add path of point to grid

    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = this.pathColor;

    for(let i = 0; i < this.path.length; i++){
      const currPos = this.path[i];
      const prevPos = this.path[i - 1];

      const pxPos = this.mToPx2(origin, currPos, ppm, scale);
      const prevPxPos = prevPos ? this.mToPx2(origin, prevPos, ppm, scale) : null;

      const rico = Vector.rico(prevPxPos, pxPos);

      if(this.isPointOnGrid2(pxPos, gridDim)){
        currPos.isOnGrid = true;
        if(prevPos){
          if(!prevPos.isOnGrid){
            const tempPxPos = this.getPrevPointOnGrid(pxPos, prevPxPos.copy(), gridDim);
            ctx.moveTo(tempPxPos.x, -tempPxPos.y);
          }
          ctx.lineTo(pxPos.x, -pxPos.y);
        } else {
          ctx.moveTo(pxPos.x, -pxPos.y);
        }
      } else {
        currPos.isOnGrid = false;

        if(prevPos){
          if(prevPos && !prevPos.isOnGrid){
            const tempPxPos = this.getPrevPointOnGrid(pxPos, prevPxPos.copy(), gridDim);
            ctx.moveTo(tempPxPos.x, -tempPxPos.y);
          }
  
          if(pxPos.x < 0){
            if(prevPxPos.x < 0) continue;
            pxPos.y = pxPos.y + pxPos.x * -rico;
            pxPos.x = 0;
          }
  
          if(pxPos.x > gridDim.x){
            if(prevPxPos.x > gridDim.x) continue;
            pxPos.y = prevPxPos.y + (prevPxPos.x - gridDim.x) * -rico;
            pxPos.x = gridDim.x;
          }
  
          if(pxPos.y < 0){
            if(prevPxPos.y < 0) continue;
            pxPos.x = prevPxPos.x - prevPxPos.y / rico;
            pxPos.y = 0;
          }
  
          if(pxPos.y > gridDim.y){
            if(prevPxPos.y > gridDim.y) continue;
            pxPos.x = prevPxPos.x + (gridDim.y - prevPxPos.y) / rico;
            pxPos.y = gridDim.y;
          }
        }

        if(prevPos){
          ctx.lineTo(pxPos.x, -pxPos.y);
        } else {
          ctx.moveTo(pxPos.x, -pxPos.y);
        }
      }
    }

    ctx.stroke();
    ctx.closePath();
  }
}

const canv = new Canvas();

const grid = new Grid();
grid.addPoint(1, 1, 3, 12);

let time = 0;
const maxtime = 2;

const realtime = true;
let lastTime = new Date().getTime();

let dt = 0.5;
let rdt = dt;

function animate() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const now = new Date().getTime()
  rdt = (now - lastTime) / 1000;
  lastTime = now;

  if(realtime){
    dt = rdt;
  }

  const mouse = canv.getMouse();
  grid.resize(canvas.width, canvas.height);
  grid.updateMouse(mouse.pos, mouse.isMouseDown, rdt);
  grid.updateOrigin(mouse.isMouseDown);

  if(time < maxtime){
    grid.update(dt);

    time += dt;
  }

  grid.render();

  requestAnimationFrame(animate);
}

animate();