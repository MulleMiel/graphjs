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

    canvas.addEventListener('mousemove', (e) => { this.mouseMove(e); }, false);
    canvas.addEventListener('mousedown', (e) => { this.mouseDown(e); }, false);
    canvas.addEventListener('mouseup', (e) => { this.mouseUp(e); }, false);
  }

  mouseMove(e){
    const pos = getMousePos(e);
    this.mouseX = pos.x;
    this.mouseY = pos.y;
  }

  mouseDown(e){
    this.isMouseDown = true;
  }

  mouseUp(e){
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
    this.originX = 200;
    this.originY = 200;
    this.scaleX = 1;
    this.scaleY = 1;
    this.ppm = 100;
    this.spacing1 = 100;
    this.spacing2 = 10;
    this.color1 = "#999999";
    this.color2 = "#dddddd";
    this.color3 = "#000000";
    this.textColor = "#000000";
    this.axisTextOffset = 20;

    this.lastDragX = null;
    this.lastDragY = null; 

    this.points = [];
    this.lines = [];
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
      this.points[i].render(this.ppm, this.scaleX, this.scaleY);
    }
  }

  drag(x, y, active) {
    if(active){
      if(this.lastDragX !== null){
        const deltaX = x - this.lastDragX;
        const deltaY = y - this.lastDragY;
        this.originX += deltaX;
        this.originY -= deltaY;
      }
      this.lastDragX = x;
      this.lastDragY = y; 
    } else {
      this.lastDragX = null;
      this.lastDragY = null; 
    }
  }

  update(dt) {
    this.updatePoints(dt);
  }

  render(){

    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';

    ctx.translate(this.originX, -this.originY);
    
    // Horizontal lines
    let a = -this.offsetY / this.spacing1;
    for(let y = canvas.height + this.offsetY; y > this.originY; y -= this.spacing2){
      ctx.beginPath();
      ctx.moveTo(-this.offsetX, y);
      ctx.lineTo(canvas.width, y);
      ctx.strokeStyle = this.color2;
      if((canvas.height - y) % this.spacing1 === 0){
        ctx.strokeStyle = this.color1;
        ctx.fillStyle = this.textColor;
        //console.log(-this.axisTextOffset - this.offsetX);
        ctx.fillText(a / this.scaleX, -this.axisTextOffset - this.offsetX, y + 5);
        if(a === 0){
          ctx.strokeStyle = this.color3;
        }
        a++;
      }
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.closePath();
    }

    //console.log(-this.offsetY / this.spacing1, a - 1);



    // Vertical lines
    let b = -this.offsetX / this.spacing1;
    for(let x = -this.offsetX; x < canvas.width; x += this.spacing2){
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height + this.offsetY);
      ctx.strokeStyle = this.color2;
      if(x % this.spacing1 === 0){
        ctx.strokeStyle = this.color1;
        ctx.fillStyle = this.textColor;
        ctx.fillText(b / this.scaleY, x, canvas.height + this.axisTextOffset + this.offsetY);
        if(b === 0){
          ctx.strokeStyle = this.color3;
        }
        b++;
      }
      
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.closePath();
    }

    this.renderPoints();

    ctx.translate(-this.originX, this.originY);
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
  }

  render(ppm, scaleX, scaleY) {
    ctx.beginPath();
    ctx.fillStyle = this.color;
    ctx.arc(this.pos.x * ppm * scaleX, canvas.height - this.pos.y * ppm * scaleY, this.radius, 0, Math.PI * 2);
    this.path.push(new Vector(this.pos.x, this.pos.y));
    ctx.fill();
    ctx.closePath();

    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = this.pathColor;
    ctx.moveTo(this.path[0].x * ppm * scaleX, canvas.height - this.path[0].y * ppm * scaleY);
    for(let i = 1; i < this.path.length; i++){
      ctx.lineTo(this.path[i].x * ppm * scaleX, canvas.height -  this.path[i].y * ppm * scaleY);
    }

    ctx.stroke();
    ctx.closePath();
  }
}

const canv = new Canvas();

const grid = new Grid();
grid.addPoint(1, 1, 4, 12);

let time = 0;
const maxtime = 2.5;

let lastTime = new Date().getTime();

function animate() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const now = new Date().getTime()
  const dt = (now - lastTime) / 1000;
  lastTime = now;

  const mouse = canv.getMouse();
  grid.drag(mouse.x, mouse.y, mouse.isMouseDown);

  if(time < maxtime){
    grid.update(dt);

    time += dt;
  }

  grid.render();

  requestAnimationFrame(animate);
}

animate();