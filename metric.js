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
    this.mouseX = 0;
    this.mouseY = 0;
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
    this.mouseX = pos.x;
    this.mouseY = pos.y;
  }

  mouseDown(){
    this.isMouseDown = true;
  }

  mouseUp(){
    this.isMouseDown = false;
  }

  getMouse(){
    return {
      x: this.mouseX,
      y: this.mouseY,
      isMouseDown: this.isMouseDown
    }
  }
}

class Grid {
  constructor(){
    this.offsetX = 100;
    this.offsetY = 100;
    this.gridWidth = 300;
    this.gridHeight = 300;
    this.originX = 200;
    this.originY = 200;
    this.zoomSteps = 10;
    this.zoomStep = 5;
    this.scaleStep = 0;
    this.scaleX = 1;
    this.scaleY = 1;
    this.ppm = 62.5; // px per meter
    this.spacing1 = 100;
    this.color1 = "#999999";
    this.color2 = "#dddddd";
    this.color3 = "#000000";
    this.textColor = "#000000";
    this.axisTextOffset = 20;

    this.mouseX = null;
    this.mouseY = null; 
    this.lastDragX = null;
    this.lastDragY = null;

    this.velocityX = 0;
    this.velocityY = 0;
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

  applyZoom(deltaScaleX, deltaScaleY, withMouse){
    const zoomFactor = 1 + this.zoomStep / this.zoomSteps;
    const prevPpm = this.ppm;
    this.ppm = this.spacing1 * zoomFactor;

    if(this.mouseX !== null){
      const mousePxX = withMouse ? // Zoom to:
        this.mouseX - this.originX - this.offsetX : // Center off mouse X
        -this.originX + this.gridWidth / 2; // Center off canvas width

      const mousePxY = withMouse ?  // Zoom to:
        this.gridHeight - this.originY - this.mouseY : // Center off mouse Y
        -this.originY + this.gridHeight / 2; // Center off height

      if(deltaScaleX === 0){
        const factor = 1 + (this.ppm - prevPpm) / prevPpm;
        const deltaMousePxX = mousePxX - mousePxX * factor;
        const deltaMousePxY = mousePxY - mousePxY * factor;

        this.originX += deltaMousePxX;
        this.originY += deltaMousePxY;
      } else {
        const oldXValue = mousePxX / prevPpm / (this.scaleX - deltaScaleX);
        const newXValue = mousePxX / this.ppm / this.scaleX;
        const deltaXPx = (newXValue - oldXValue) * this.ppm * this.scaleX;
        this.originX += deltaXPx;

        const oldYValue = mousePxY / prevPpm / (this.scaleY - deltaScaleY);
        const newYValue = mousePxY / this.ppm / this.scaleY;
        const deltaYPx = (newYValue - oldYValue) * this.ppm * this.scaleY;
        this.originY += deltaYPx;
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

    const deltaScaleX = order - this.scaleX;
    const deltaScaleY = order - this.scaleY;

    this.scaleX = order;
    this.scaleY = order;

    this.applyZoom(deltaScaleX, deltaScaleY, withMouse);
  }

  resetZoom(){
    this.zoomStep = 5;
    this.scaleStep = 0;
    this.scaleX = 1;
    this.scaleY = 1;
    this.originX = 0;
    this.originY = 0;
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
      this.points[i].render(this.originX, this.originY, this.gridWidth, this.gridHeight, this.ppm, this.scaleX, this.scaleY);
    }
  }

  resize(width, height) {
    this.gridWidth = width - this.offsetX;
    this.gridHeight = height - this.offsetY;
  }

  updateMouse(x, y, active, dt) {
    this.mouseX = x;
    this.mouseY = y;

    let velocityX = 0;
    let velocityY = 0;

    if(active){
      if(this.lastDragX !== null){
        const deltaX = this.mouseX - this.lastDragX;
        const deltaY = this.mouseY - this.lastDragY;
        this.originX += deltaX;
        this.originY -= deltaY;

        velocityX = deltaX / dt; // px/s
        velocityY = -deltaY / dt; // px/s

        this.velocityX = velocityX;
        this.velocityY = velocityY;
      }
      this.lastDragX = this.mouseX;
      this.lastDragY = this.mouseY; 
    } else {
      this.lastDragX = null;
      this.lastDragY = null; 
    }
  }

  updateOrigin(isMouseDown) {
    const velocityReduceConstant = 200;
    this.velocityX -= this.velocityX * this.friction;
    this.velocityY -= this.velocityY * this.friction;
    if(!isMouseDown){
      this.originX += this.velocityX / velocityReduceConstant;
      this.originY += this.velocityY / velocityReduceConstant;
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

    ctx.translate(this.offsetX, this.gridHeight);

    const spacing = this.ppm;

    // Positive horizontal grid lines
    let y = this.originY;
    if(y > 0 && y < this.gridHeight){
      ctx.fillText(0, -this.axisTextOffset, -y + 5);
      this.renderLine([0, -y], [this.gridWidth, -y], this.color3, 1);
    }

    let i = this.originY < 0 ? 
    Math.floor(-this.originY / (spacing / 10)) : 1;

    while(y < this.gridHeight) {
      y = this.originY + i * spacing / 10;
      if(y < 0) {
        i++;
        continue
      }
      if(i % 10 === 0){
        this.renderLine([0, -y], [this.gridWidth, -y], this.color1, 1);
        const axisNumber = (y - this.originY) / spacing / this.scaleY;
        this.formatAxisNumber(axisNumber, -this.axisTextOffset, -y, true);
      } else {
        this.renderLine([0, -y], [this.gridWidth, -y], this.color2, 1);
      }
      i++;
    }

    // Negative horizontal grid lines
    i = this.originY > this.gridHeight ? 
    Math.floor((this.originY - this.gridHeight) / (spacing / 10)) : 1;

    y = this.originY - i * spacing / 10;
    while(y > 0) {
      if(i % 10 === 0){
        this.renderLine([0, -y], [this.gridWidth, -y], this.color1, 1);
        const axisNumber = (y - this.originY) / spacing / this.scaleY;
        this.formatAxisNumber(axisNumber, -this.axisTextOffset, -y, true);
      } else {
        this.renderLine([0, -y], [this.gridWidth, -y], this.color2, 1);
      }
      i++;
      y = this.originY - i * spacing / 10;
    } 

    ctx.textAlign = 'center';

    // Positive vertical  grid lines
    let x = this.originX;
    if(x > 0 && x < this.gridWidth){
      ctx.fillText(0, x, this.axisTextOffset);
      this.renderLine([x, 0], [x, -this.gridHeight], this.color3, 1);
    }

    i = this.originX < 0 ? 
    Math.floor(-this.originX / (spacing / 10)) : 1;

    while(x < this.gridWidth) {
      x = this.originX + i * spacing / 10;
      
      if(x < 0) {
        i++;
        continue;
      }
      if(i % 10 === 0){
        this.renderLine([x, 0], [x, -this.gridHeight], this.color1, 1);
        const axisNumber = (x - this.originX) / spacing / this.scaleX;
        this.formatAxisNumber(axisNumber, x, this.axisTextOffset, false);
      } else {
        this.renderLine([x, 0], [x, -this.gridHeight], this.color2, 1);
      }
      i++;
    }

    // Negative vertical  grid lines
    i = this.originX > this.gridWidth ? 
    Math.floor(-(this.gridWidth - this.originX) / (spacing / 10)) : 1;

    x = this.originX - i * spacing / 10;
    while(x > 0) {
      if(i % 10 === 0){
        this.renderLine([x, 0], [x, -this.gridHeight], this.color1, 1);
        const axisNumber = (x - this.originX) / spacing / this.scaleX;
        this.formatAxisNumber(axisNumber, x, this.axisTextOffset, false);
      } else {
        this.renderLine([x, 0], [x, -this.gridHeight], this.color2, 1);
      }
      
      i++;
      x = this.originX - i * spacing / 10;
    }

    this.renderPoints();

    ctx.translate(-this.offsetX, -this.gridHeight);
  }
}

class Point {
  
  constructor(x, y, vx, vy) {
    this.pos = new Vector(x, y);
    this.vel = new Vector(vx, vy);

    this.gravity = new Vector(0, -9.81);

    this.path = [new Vector(x, y)];
    this.pathColor = '#e62a4f';
    
    this.radius = 10;
    this.color = '#e62a4f';
    this.mass = 1;
  }
  
  update(dt) {
    const deltaVel = new Vector(this.gravity.x * dt, this.gravity.y * dt);
    this.vel.add(deltaVel);
    const deltaPos = new Vector(this.vel.x * dt, this.vel.y * dt);
    this.pos.add(deltaPos);

    this.path.push(new Vector(this.pos.x, this.pos.y));
  }

  getPrevPointOnGrid(prevPxPosX, prevPxPosY, pxPosX, pxPosY, gridWidth, gridHeight){

    const ricoToPrevious = (pxPosY - prevPxPosY) / (pxPosX - prevPxPosX);

    if(pxPosX < 0){
      //pxPosY = pxPosY + pxPosX * -ricoToNext;
      pxPosY = pxPosY + pxPosX * -ricoToPrevious;
      pxPosX = 0;
    }

    if(pxPosX > gridWidth){
      pxPosY = prevPxPosY + (prevPxPosX - gridWidth) * -ricoToPrevious;
      pxPosX = gridWidth;
    }

    if(pxPosY > 0){
      pxPosX = prevPxPosX - (prevPxPosY) / ricoToPrevious;
      pxPosY = 0;
    }

    if(pxPosY < -gridHeight){
      pxPosX = prevPxPosX - (prevPxPosY + gridHeight) / ricoToPrevious;
      pxPosY = -gridHeight;
    }

    return {
      x: pxPosX,
      y: pxPosY
    }
  }

  isPointOnGrid(x, y, width, height){
    if(x >= 0 && x < width &&
      y <= 0 && y > -height){
      return true;
    }
    return false;
  }

  mToPx(origin, pos, ppm, scale){
    return origin + pos * ppm * scale;
  }

  render(originX, originY, gridWidth, gridHeight, ppm, scaleX, scaleY) {

    // Add circle to grid

    // Convert metric position to pixel position (meter to pixel)
    let pxPosX = this.mToPx(originX, this.pos.x, ppm, scaleX);
    let pxPosY = this.mToPx(-originY, -this.pos.y, ppm, scaleY);

    if(this.isPointOnGrid(pxPosX, pxPosY, gridWidth, gridHeight)){
      ctx.beginPath();
      ctx.fillStyle = this.color;
      ctx.arc(pxPosX, pxPosY, this.radius, 0, Math.PI * 2);
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
      const nextPos = this.path[i + 1];

      pxPosX = this.mToPx(originX, currPos.x, ppm, scaleX);
      pxPosY = this.mToPx(-originY, -currPos.y, ppm, scaleY);

      const prevPxPosX = prevPos ? this.mToPx(originX, prevPos.x, ppm, scaleX) : null;
      const prevPxPosY = prevPos ? this.mToPx(-originY, -prevPos.y, ppm, scaleY) : null;
      const nextPxPosX = nextPos ? this.mToPx(originX, nextPos.x, ppm, scaleX) : null;
      const nextPxPosY = nextPos ? this.mToPx(-originY, -nextPos.y, ppm, scaleY) : null;

      const ricoToNext = (nextPxPosY - pxPosY) / (nextPxPosX - pxPosX);
      const ricoToPrevious = (pxPosY - prevPxPosY) / (pxPosX - prevPxPosX);

      if(this.isPointOnGrid(pxPosX, pxPosY, gridWidth, gridHeight)){
        currPos.isOnGrid = true;
        if(prevPos){
          if(!prevPos.isOnGrid){
            const tempPxPos = this.getPrevPointOnGrid(pxPosX, pxPosY, prevPxPosX, prevPxPosY, gridWidth, gridHeight);
            ctx.moveTo(tempPxPos.x, tempPxPos.y);
          }
          ctx.lineTo(pxPosX, pxPosY);
        } else {
          ctx.moveTo(pxPosX, pxPosY);
        }
      } else {
        currPos.isOnGrid = false;

        if(prevPos && !prevPos.isOnGrid){
          const tempPxPos = this.getPrevPointOnGrid(pxPosX, pxPosY, prevPxPosX, prevPxPosY, gridWidth, gridHeight);
          ctx.moveTo(tempPxPos.x, tempPxPos.y);
        }

        if(pxPosX < 0){
          if(prevPxPosX < 0) continue;
          pxPosY = pxPosY + pxPosX * -ricoToPrevious;
          pxPosX = 0;
        }

        if(pxPosX > gridWidth){
          if(prevPxPosX > gridWidth) continue;
          pxPosY = prevPxPosY + (prevPxPosX - gridWidth) * -ricoToPrevious;
          pxPosX = gridWidth;
        }

        if(pxPosY > 0){
          if(prevPxPosY > 0) continue;
          pxPosX = prevPxPosX - prevPxPosY / ricoToPrevious;
          pxPosY = 0;
        }

        if(pxPosY < -gridHeight){
          if(prevPxPosY < -gridHeight) continue;
            pxPosX = prevPxPosX - (prevPxPosY + gridHeight) / ricoToPrevious;
            pxPosY = -gridHeight;
        }

        if(prevPos){
          ctx.lineTo(pxPosX, pxPosY);
        } else {
          ctx.moveTo(pxPosX, pxPosY);
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

let dt = 0.1;
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
  grid.updateMouse(mouse.x, mouse.y, mouse.isMouseDown, rdt);
  grid.updateOrigin(mouse.isMouseDown);

  if(time < maxtime){
    grid.update(dt);

    time += dt;
  }

  grid.render();

  requestAnimationFrame(animate);
}

animate();