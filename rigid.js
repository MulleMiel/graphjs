const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const NUM_RIGID_BODIES = 1;

class BoxShape {
  constructor(width, height, depth, density = 7850){
    this.density = density; // kg/m³ (default steel)
    this.width = width; // m
    this.height = height; // m
    this.depth = depth; // m
    this.mass = this.calcMass(); // kg
    this.momentOfInertia = this.calcInertia();
  }

  calcMass(){
    return this.width * this.height * this.depth * this.density;
  }

  calcInertia(){
    return this.mass * (this.width * this.width + 
      this.height * this.height) / 12;
  }
}

class RigidBody {
  constructor(
           position = [0, 0], 
     linearVelocity = [0, 0], 
              angle = 0, 
    angularVelocity = 0,
              force = [0, 0],
             moment = 0,
              shape){

    this.position = position;
    this.linearVelocity = linearVelocity;
    this.angle = position;
    this.linearVelocity = linearVelocity;
    this.angle = angle;
    this.angularVelocity = angularVelocity;
    this.force = force;
    this.moment = moment;
    this.shape = shape;
  }

  draw() {
    //draw a circle
    const scale = 100;
    const x = this.position[0] * scale;
    const y = this.position[1] * scale;
    const width = this.shape.width * scale;
    const height = this.shape.height * scale;
    console.log(x, y, width, height, x + width / 2);

    ctx.translate(x + width / 2, y + height / 2);
    ctx.fillRect(400, 400, width, height);
    ctx.rotate(1 * Math.PI / 180);
    //ctx.translate(x + width / 2 * -1, y + height / 2 * -1);
  }
}

const rigidBodies = [];
for(let i = 0; i < NUM_RIGID_BODIES; i++) {
  rigidBodies.push(new RigidBody([4, 8]));
}

function initializeRigidBodies() {
  for (const rigidBody of rigidBodies) {
    const shape = new BoxShape(0.5, 0.1, 0.1);
    rigidBody.shape = shape;
  }
}

function printRigidBodies(){
  for (const rigidBody of rigidBodies) {
    rigidBody.draw();
  }
}

function computeForceAndTorque(rigidBody) {
  const force = [0, 0];

  rigidBody.force = force;
  // r is the 'arm vector' that goes from the center of mass to the point of force application
  const arm = [rigidBody.shape.width / 2, rigidBody.shape.height / 2];
  rigidBody.moment = arm[0] * force[1] - arm[1] * force[0];
}

initializeRigidBodies();
printRigidBodies();

const dt = 0.01;

function animate() {
  
  ctx.restore();

  for (const rigidBody of rigidBodies) {
    computeForceAndTorque(rigidBody);
    const linearAcceleration = [
      rigidBody.force[0] / rigidBody.shape.mass, 
      rigidBody.force[1] / rigidBody.shape.mass
    ];
    rigidBody.linearVelocity[0] += linearAcceleration[0] * dt;
    rigidBody.linearVelocity[1] += linearAcceleration[1] * dt;
    rigidBody.position[0] += rigidBody.linearVelocity[0] * dt;
    rigidBody.position[1] += rigidBody.linearVelocity[1] * dt;
    
    const angularAcceleration = rigidBody.moment / rigidBody.shape.momentOfInertia;
    rigidBody.angularVelocity += angularAcceleration * dt;
    rigidBody.angle += rigidBody.angularVelocity * dt;
  }

  for (const rigidBody of rigidBodies) {
    rigidBody.draw();
  }

  //window.requestAnimationFrame(animate);
}

window.requestAnimationFrame(animate);