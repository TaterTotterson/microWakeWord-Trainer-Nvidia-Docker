//#region node_modules/@vue/shared/dist/shared.esm-bundler.js
// @__NO_SIDE_EFFECTS__
function e(e) {
	let t = /* @__PURE__ */ Object.create(null);
	for (let n of e.split(",")) t[n] = 1;
	return (e) => e in t;
}
var t = {}, n = [], r = () => {}, i = () => !1, a = (e) => e.charCodeAt(0) === 111 && e.charCodeAt(1) === 110 && (e.charCodeAt(2) > 122 || e.charCodeAt(2) < 97), o = (e) => e.startsWith("onUpdate:"), s = Object.assign, c = (e, t) => {
	let n = e.indexOf(t);
	n > -1 && e.splice(n, 1);
}, l = Object.prototype.hasOwnProperty, u = (e, t) => l.call(e, t), d = Array.isArray, f = (e) => x(e) === "[object Map]", p = (e) => x(e) === "[object Set]", m = (e) => x(e) === "[object Date]", h = (e) => typeof e == "function", g = (e) => typeof e == "string", _ = (e) => typeof e == "symbol", v = (e) => typeof e == "object" && !!e, y = (e) => (v(e) || h(e)) && h(e.then) && h(e.catch), b = Object.prototype.toString, x = (e) => b.call(e), S = (e) => x(e).slice(8, -1), C = (e) => x(e) === "[object Object]", w = (e) => g(e) && e !== "NaN" && e[0] !== "-" && "" + parseInt(e, 10) === e, ee = /* @__PURE__ */ e(",key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"), te = (e) => {
	let t = /* @__PURE__ */ Object.create(null);
	return ((n) => t[n] || (t[n] = e(n)));
}, ne = /-\w/g, T = te((e) => e.replace(ne, (e) => e.slice(1).toUpperCase())), re = /\B([A-Z])/g, E = te((e) => e.replace(re, "-$1").toLowerCase()), ie = te((e) => e.charAt(0).toUpperCase() + e.slice(1)), ae = te((e) => e ? `on${ie(e)}` : ""), D = (e, t) => !Object.is(e, t), oe = (e, ...t) => {
	for (let n = 0; n < e.length; n++) e[n](...t);
}, O = (e, t, n, r = !1) => {
	Object.defineProperty(e, t, {
		configurable: !0,
		enumerable: !1,
		writable: r,
		value: n
	});
}, se = (e) => {
	let t = parseFloat(e);
	return isNaN(t) ? e : t;
}, ce = (e) => {
	let t = g(e) ? Number(e) : NaN;
	return isNaN(t) ? e : t;
}, le, ue = () => le ||= typeof globalThis < "u" ? globalThis : typeof self < "u" ? self : typeof window < "u" ? window : typeof global < "u" ? global : {};
function de(e) {
	if (d(e)) {
		let t = {};
		for (let n = 0; n < e.length; n++) {
			let r = e[n], i = g(r) ? he(r) : de(r);
			if (i) for (let e in i) t[e] = i[e];
		}
		return t;
	}
	if (g(e) || v(e)) return e;
}
var fe = /;(?![^(]*\))/g, pe = /:([^]+)/, me = /\/\*[^]*?\*\//g;
function he(e) {
	let t = {};
	return e.replace(me, "").split(fe).forEach((e) => {
		if (e) {
			let n = e.split(pe);
			n.length > 1 && (t[n[0].trim()] = n[1].trim());
		}
	}), t;
}
function k(e) {
	let t = "";
	if (g(e)) t = e;
	else if (d(e)) for (let n = 0; n < e.length; n++) {
		let r = k(e[n]);
		r && (t += r + " ");
	}
	else if (v(e)) for (let n in e) e[n] && (t += n + " ");
	return t.trim();
}
var ge = "itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly", _e = /* @__PURE__ */ e(ge);
ge + "";
function ve(e) {
	return !!e || e === "";
}
function ye(e, t) {
	if (e.length !== t.length) return !1;
	let n = !0;
	for (let r = 0; n && r < e.length; r++) n = be(e[r], t[r]);
	return n;
}
function be(e, t) {
	if (e === t) return !0;
	let n = m(e), r = m(t);
	if (n || r) return n && r ? e.getTime() === t.getTime() : !1;
	if (n = _(e), r = _(t), n || r) return e === t;
	if (n = d(e), r = d(t), n || r) return n && r ? ye(e, t) : !1;
	if (n = v(e), r = v(t), n || r) {
		if (!n || !r || Object.keys(e).length !== Object.keys(t).length) return !1;
		for (let n in e) {
			let r = e.hasOwnProperty(n), i = t.hasOwnProperty(n);
			if (r && !i || !r && i || !be(e[n], t[n])) return !1;
		}
	}
	return String(e) === String(t);
}
function xe(e, t) {
	return e.findIndex((e) => be(e, t));
}
var Se = (e) => !!(e && e.__v_isRef === !0), A = (e) => g(e) ? e : e == null ? "" : d(e) || v(e) && (e.toString === b || !h(e.toString)) ? Se(e) ? A(e.value) : JSON.stringify(e, Ce, 2) : String(e), Ce = (e, t) => Se(t) ? Ce(e, t.value) : f(t) ? { [`Map(${t.size})`]: [...t.entries()].reduce((e, [t, n], r) => (e[we(t, r) + " =>"] = n, e), {}) } : p(t) ? { [`Set(${t.size})`]: [...t.values()].map((e) => we(e)) } : _(t) ? we(t) : v(t) && !d(t) && !C(t) ? String(t) : t, we = (e, t = "") => _(e) ? `Symbol(${e.description ?? t})` : e, j, Te = class {
	constructor(e = !1) {
		this.detached = e, this._active = !0, this._on = 0, this.effects = [], this.cleanups = [], this._isPaused = !1, this._warnOnRun = !0, this.__v_skip = !0, !e && j && (j.active ? (this.parent = j, this.index = (j.scopes || (j.scopes = [])).push(this) - 1) : (this._active = !1, this._warnOnRun = !1));
	}
	get active() {
		return this._active;
	}
	pause() {
		if (this._active) {
			this._isPaused = !0;
			let e, t;
			if (this.scopes) {
				let n = this.scopes.slice();
				for (e = 0, t = n.length; e < t; e++) n[e].pause();
			}
			for (e = 0, t = this.effects.length; e < t; e++) this.effects[e].pause();
		}
	}
	resume() {
		if (this._active && this._isPaused) {
			this._isPaused = !1;
			let e, t;
			if (this.scopes) {
				let n = this.scopes.slice();
				for (e = 0, t = n.length; e < t; e++) n[e].resume();
			}
			let n = this.effects.slice();
			for (e = 0, t = n.length; e < t; e++) n[e].resume();
		}
	}
	run(e) {
		if (this._active) {
			let t = j;
			try {
				return j = this, e();
			} finally {
				j = t;
			}
		}
	}
	on() {
		++this._on === 1 && (this.prevScope = j, j = this);
	}
	off() {
		if (this._on > 0 && --this._on === 0) {
			if (j === this) j = this.prevScope;
			else {
				let e = j;
				for (; e;) {
					if (e.prevScope === this) {
						e.prevScope = this.prevScope;
						break;
					}
					e = e.prevScope;
				}
			}
			this.prevScope = void 0;
		}
	}
	stop(e) {
		if (this._active) {
			this._active = !1;
			let t, n;
			for (t = 0, n = this.effects.length; t < n; t++) this.effects[t].stop();
			for (this.effects.length = 0, t = 0, n = this.cleanups.length; t < n; t++) this.cleanups[t]();
			if (this.cleanups.length = 0, this.scopes) {
				let e = this.scopes.slice();
				for (t = 0, n = e.length; t < n; t++) e[t].stop(!0);
				this.scopes.length = 0;
			}
			if (!this.detached && this.parent && !e) {
				let e = this.parent.scopes.pop();
				e && e !== this && (this.parent.scopes[this.index] = e, e.index = this.index);
			}
			this.parent = void 0;
		}
	}
};
function Ee() {
	return j;
}
var M, De = /* @__PURE__ */ new WeakSet(), Oe = class {
	constructor(e) {
		this.fn = e, this.deps = void 0, this.depsTail = void 0, this.flags = 5, this.next = void 0, this.cleanup = void 0, this.scheduler = void 0, j && (j.active ? j.effects.push(this) : this.flags &= -2);
	}
	pause() {
		this.flags |= 64;
	}
	resume() {
		this.flags & 64 && (this.flags &= -65, De.has(this) && (De.delete(this), this.trigger()));
	}
	notify() {
		this.flags & 2 && !(this.flags & 32) || this.flags & 8 || Me(this);
	}
	run() {
		if (!(this.flags & 1)) return this.fn();
		this.flags |= 2, Ge(this), Fe(this);
		let e = M, t = Ve;
		M = this, Ve = !0;
		try {
			return this.fn();
		} finally {
			Ie(this), M = e, Ve = t, this.flags &= -3;
		}
	}
	stop() {
		if (this.flags & 1) {
			for (let e = this.deps; e; e = e.nextDep) ze(e);
			this.deps = this.depsTail = void 0, Ge(this), this.onStop && this.onStop(), this.flags &= -2;
		}
	}
	trigger() {
		this.flags & 64 ? De.add(this) : this.scheduler ? this.scheduler() : this.runIfDirty();
	}
	runIfDirty() {
		Le(this) && this.run();
	}
	get dirty() {
		return Le(this);
	}
}, ke = 0, Ae, je;
function Me(e, t = !1) {
	if (e.flags |= 8, t) {
		e.next = je, je = e;
		return;
	}
	e.next = Ae, Ae = e;
}
function Ne() {
	ke++;
}
function Pe() {
	if (--ke > 0) return;
	if (je) {
		let e = je;
		for (je = void 0; e;) {
			let t = e.next;
			e.next = void 0, e.flags &= -9, e = t;
		}
	}
	let e;
	for (; Ae;) {
		let t = Ae;
		for (Ae = void 0; t;) {
			let n = t.next;
			if (t.next = void 0, t.flags &= -9, t.flags & 1) try {
				t.trigger();
			} catch (t) {
				e ||= t;
			}
			t = n;
		}
	}
	if (e) throw e;
}
function Fe(e) {
	for (let t = e.deps; t; t = t.nextDep) t.version = -1, t.prevActiveLink = t.dep.activeLink, t.dep.activeLink = t;
}
function Ie(e) {
	let t, n = e.depsTail, r = n;
	for (; r;) {
		let e = r.prevDep;
		r.version === -1 ? (r === n && (n = e), ze(r), Be(r)) : t = r, r.dep.activeLink = r.prevActiveLink, r.prevActiveLink = void 0, r = e;
	}
	e.deps = t, e.depsTail = n;
}
function Le(e) {
	for (let t = e.deps; t; t = t.nextDep) if (t.dep.version !== t.version || t.dep.computed && (Re(t.dep.computed) || t.dep.version !== t.version)) return !0;
	return !!e._dirty;
}
function Re(e) {
	if (e.flags & 4 && !(e.flags & 16) || (e.flags &= -17, e.globalVersion === Ke) || (e.globalVersion = Ke, !e.isSSR && e.flags & 128 && (!e.deps && !e._dirty || !Le(e)))) return;
	e.flags |= 2;
	let t = e.dep, n = M, r = Ve;
	M = e, Ve = !0;
	try {
		Fe(e);
		let n = e.fn(e._value);
		(t.version === 0 || D(n, e._value)) && (e.flags |= 128, e._value = n, t.version++);
	} catch (e) {
		throw t.version++, e;
	} finally {
		M = n, Ve = r, Ie(e), e.flags &= -3;
	}
}
function ze(e, t = !1) {
	let { dep: n, prevSub: r, nextSub: i } = e;
	if (r && (r.nextSub = i, e.prevSub = void 0), i && (i.prevSub = r, e.nextSub = void 0), n.subs === e && (n.subs = r, !r && n.computed)) {
		n.computed.flags &= -5;
		for (let e = n.computed.deps; e; e = e.nextDep) ze(e, !0);
	}
	!t && !--n.sc && n.map && n.map.delete(n.key);
}
function Be(e) {
	let { prevDep: t, nextDep: n } = e;
	t && (t.nextDep = n, e.prevDep = void 0), n && (n.prevDep = t, e.nextDep = void 0);
}
var Ve = !0, He = [];
function Ue() {
	He.push(Ve), Ve = !1;
}
function We() {
	let e = He.pop();
	Ve = e === void 0 || e;
}
function Ge(e) {
	let { cleanup: t } = e;
	if (e.cleanup = void 0, t) {
		let e = M;
		M = void 0;
		try {
			t();
		} finally {
			M = e;
		}
	}
}
var Ke = 0, qe = class {
	constructor(e, t) {
		this.sub = e, this.dep = t, this.version = t.version, this.nextDep = this.prevDep = this.nextSub = this.prevSub = this.prevActiveLink = void 0;
	}
}, Je = class {
	constructor(e) {
		this.computed = e, this.version = 0, this.activeLink = void 0, this.subs = void 0, this.map = void 0, this.key = void 0, this.sc = 0, this.__v_skip = !0;
	}
	track(e) {
		if (!M || !Ve || M === this.computed) return;
		let t = this.activeLink;
		if (t === void 0 || t.sub !== M) t = this.activeLink = new qe(M, this), M.deps ? (t.prevDep = M.depsTail, M.depsTail.nextDep = t, M.depsTail = t) : M.deps = M.depsTail = t, Ye(t);
		else if (t.version === -1 && (t.version = this.version, t.nextDep)) {
			let e = t.nextDep;
			e.prevDep = t.prevDep, t.prevDep && (t.prevDep.nextDep = e), t.prevDep = M.depsTail, t.nextDep = void 0, M.depsTail.nextDep = t, M.depsTail = t, M.deps === t && (M.deps = e);
		}
		return t;
	}
	trigger(e) {
		this.version++, Ke++, this.notify(e);
	}
	notify(e) {
		Ne();
		try {
			for (let e = this.subs; e; e = e.prevSub) e.sub.notify() && e.sub.dep.notify();
		} finally {
			Pe();
		}
	}
};
function Ye(e) {
	if (e.dep.sc++, e.sub.flags & 4) {
		let t = e.dep.computed;
		if (t && !e.dep.subs) {
			t.flags |= 20;
			for (let e = t.deps; e; e = e.nextDep) Ye(e);
		}
		let n = e.dep.subs;
		n !== e && (e.prevSub = n, n && (n.nextSub = e)), e.dep.subs = e;
	}
}
var Xe = /* @__PURE__ */ new WeakMap(), Ze = /* @__PURE__ */ Symbol(""), Qe = /* @__PURE__ */ Symbol(""), $e = /* @__PURE__ */ Symbol("");
function N(e, t, n) {
	if (Ve && M) {
		let t = Xe.get(e);
		t || Xe.set(e, t = /* @__PURE__ */ new Map());
		let r = t.get(n);
		r || (t.set(n, r = new Je()), r.map = t, r.key = n), r.track();
	}
}
function et(e, t, n, r, i, a) {
	let o = Xe.get(e);
	if (!o) {
		Ke++;
		return;
	}
	let s = (e) => {
		e && e.trigger();
	};
	if (Ne(), t === "clear") o.forEach(s);
	else {
		let i = d(e), a = i && w(n);
		if (i && n === "length") {
			let e = Number(r);
			o.forEach((t, n) => {
				(n === "length" || n === $e || !_(n) && n >= e) && s(t);
			});
		} else switch ((n !== void 0 || o.has(void 0)) && s(o.get(n)), a && s(o.get($e)), t) {
			case "add":
				i ? a && s(o.get("length")) : (s(o.get(Ze)), f(e) && s(o.get(Qe)));
				break;
			case "delete":
				i || (s(o.get(Ze)), f(e) && s(o.get(Qe)));
				break;
			case "set": f(e) && s(o.get(Ze));
		}
	}
	Pe();
}
function tt(e) {
	let t = /* @__PURE__ */ P(e);
	return t === e ? t : (N(t, "iterate", $e), /* @__PURE__ */ Bt(e) ? t : t.map(Ut));
}
function nt(e) {
	return N(e = /* @__PURE__ */ P(e), "iterate", $e), e;
}
function rt(e, t) {
	return /* @__PURE__ */ zt(e) ? Wt(/* @__PURE__ */ Rt(e) ? Ut(t) : t) : Ut(t);
}
var it = {
	__proto__: null,
	[Symbol.iterator]() {
		return at(this, Symbol.iterator, (e) => rt(this, e));
	},
	concat(...e) {
		return tt(this).concat(...e.map((e) => d(e) ? tt(e) : e));
	},
	entries() {
		return at(this, "entries", (e) => (e[1] = rt(this, e[1]), e));
	},
	every(e, t) {
		return st(this, "every", e, t, void 0, arguments);
	},
	filter(e, t) {
		return st(this, "filter", e, t, (e) => e.map((e) => rt(this, e)), arguments);
	},
	find(e, t) {
		return st(this, "find", e, t, (e) => rt(this, e), arguments);
	},
	findIndex(e, t) {
		return st(this, "findIndex", e, t, void 0, arguments);
	},
	findLast(e, t) {
		return st(this, "findLast", e, t, (e) => rt(this, e), arguments);
	},
	findLastIndex(e, t) {
		return st(this, "findLastIndex", e, t, void 0, arguments);
	},
	forEach(e, t) {
		return st(this, "forEach", e, t, void 0, arguments);
	},
	includes(...e) {
		return lt(this, "includes", e);
	},
	indexOf(...e) {
		return lt(this, "indexOf", e);
	},
	join(e) {
		return tt(this).join(e);
	},
	lastIndexOf(...e) {
		return lt(this, "lastIndexOf", e);
	},
	map(e, t) {
		return st(this, "map", e, t, void 0, arguments);
	},
	pop() {
		return ut(this, "pop");
	},
	push(...e) {
		return ut(this, "push", e);
	},
	reduce(e, ...t) {
		return ct(this, "reduce", e, t);
	},
	reduceRight(e, ...t) {
		return ct(this, "reduceRight", e, t);
	},
	shift() {
		return ut(this, "shift");
	},
	some(e, t) {
		return st(this, "some", e, t, void 0, arguments);
	},
	splice(...e) {
		return ut(this, "splice", e);
	},
	toReversed() {
		return tt(this).toReversed();
	},
	toSorted(e) {
		return tt(this).toSorted(e);
	},
	toSpliced(...e) {
		return tt(this).toSpliced(...e);
	},
	unshift(...e) {
		return ut(this, "unshift", e);
	},
	values() {
		return at(this, "values", (e) => rt(this, e));
	}
};
function at(e, t, n) {
	let r = nt(e), i = r[t]();
	return r !== e && !/* @__PURE__ */ Bt(e) && (i._next = i.next, i.next = () => {
		let e = i._next();
		return e.done || (e.value = n(e.value)), e;
	}), i;
}
var ot = Array.prototype;
function st(e, t, n, r, i, a) {
	let o = nt(e), s = o !== e && !/* @__PURE__ */ Bt(e), c = o[t];
	if (c !== ot[t]) {
		let t = c.apply(e, a);
		return s ? Ut(t) : t;
	}
	let l = n;
	o !== e && (s ? l = function(t, r) {
		return n.call(this, rt(e, t), r, e);
	} : n.length > 2 && (l = function(t, r) {
		return n.call(this, t, r, e);
	}));
	let u = c.call(o, l, r);
	return s && i ? i(u) : u;
}
function ct(e, t, n, r) {
	let i = nt(e), a = i !== e && !/* @__PURE__ */ Bt(e), o = n, s = !1;
	i !== e && (a ? (s = r.length === 0, o = function(t, r, i) {
		return s && (s = !1, t = rt(e, t)), n.call(this, t, rt(e, r), i, e);
	}) : n.length > 3 && (o = function(t, r, i) {
		return n.call(this, t, r, i, e);
	}));
	let c = i[t](o, ...r);
	return s ? rt(e, c) : c;
}
function lt(e, t, n) {
	let r = /* @__PURE__ */ P(e);
	N(r, "iterate", $e);
	let i = r[t](...n);
	return (i === -1 || i === !1) && /* @__PURE__ */ Vt(n[0]) ? (n[0] = /* @__PURE__ */ P(n[0]), r[t](...n)) : i;
}
function ut(e, t, n = []) {
	Ue(), Ne();
	let r = (/* @__PURE__ */ P(e))[t].apply(e, n);
	return Pe(), We(), r;
}
var dt = /* @__PURE__ */ e("__proto__,__v_isRef,__isVue"), ft = new Set(/* @__PURE__ */ Object.getOwnPropertyNames(Symbol).filter((e) => e !== "arguments" && e !== "caller").map((e) => Symbol[e]).filter(_));
function pt(e) {
	_(e) || (e = String(e));
	let t = /* @__PURE__ */ P(this);
	return N(t, "has", e), t.hasOwnProperty(e);
}
var mt = class {
	constructor(e = !1, t = !1) {
		this._isReadonly = e, this._isShallow = t;
	}
	get(e, t, n) {
		if (t === "__v_skip") return e.__v_skip;
		let r = this._isReadonly, i = this._isShallow;
		if (t === "__v_isReactive") return !r;
		if (t === "__v_isReadonly") return r;
		if (t === "__v_isShallow") return i;
		if (t === "__v_raw") return n === (r ? i ? Mt : jt : i ? At : kt).get(e) || Object.getPrototypeOf(e) === Object.getPrototypeOf(n) ? e : void 0;
		let a = d(e);
		if (!r) {
			let e;
			if (a && (e = it[t])) return e;
			if (t === "hasOwnProperty") return pt;
		}
		let o = Reflect.get(e, t, /* @__PURE__ */ F(e) ? e : n);
		if ((_(t) ? ft.has(t) : dt(t)) || (r || N(e, "get", t), i)) return o;
		if (/* @__PURE__ */ F(o)) {
			let e = a && w(t) ? o : o.value;
			return r && v(e) ? /* @__PURE__ */ It(e) : e;
		}
		return v(o) ? r ? /* @__PURE__ */ It(o) : /* @__PURE__ */ Pt(o) : o;
	}
}, ht = class extends mt {
	constructor(e = !1) {
		super(!1, e);
	}
	set(e, t, n, r) {
		let i = e[t], a = d(e) && w(t);
		if (!this._isShallow) {
			let e = /* @__PURE__ */ zt(i);
			if (!/* @__PURE__ */ Bt(n) && !/* @__PURE__ */ zt(n) && (i = /* @__PURE__ */ P(i), n = /* @__PURE__ */ P(n)), !a && /* @__PURE__ */ F(i) && !/* @__PURE__ */ F(n)) return e || (i.value = n), !0;
		}
		let o = a ? Number(t) < e.length : u(e, t), s = Reflect.set(e, t, n, /* @__PURE__ */ F(e) ? e : r);
		return e === /* @__PURE__ */ P(r) && s && (o ? D(n, i) && et(e, "set", t, n, i) : et(e, "add", t, n)), s;
	}
	deleteProperty(e, t) {
		let n = u(e, t), r = e[t], i = Reflect.deleteProperty(e, t);
		return i && n && et(e, "delete", t, void 0, r), i;
	}
	has(e, t) {
		let n = Reflect.has(e, t);
		return (!_(t) || !ft.has(t)) && N(e, "has", t), n;
	}
	ownKeys(e) {
		return N(e, "iterate", d(e) ? "length" : Ze), Reflect.ownKeys(e);
	}
}, gt = class extends mt {
	constructor(e = !1) {
		super(!0, e);
	}
	set(e, t) {
		return !0;
	}
	deleteProperty(e, t) {
		return !0;
	}
}, _t = /* @__PURE__ */ new ht(), vt = /* @__PURE__ */ new gt(), yt = /* @__PURE__ */ new ht(!0), bt = (e) => e, xt = (e) => Reflect.getPrototypeOf(e);
function St(e, t, n) {
	return function(...r) {
		let i = this.__v_raw, a = /* @__PURE__ */ P(i), o = f(a), c = e === "entries" || e === Symbol.iterator && o, l = e === "keys" && o, u = i[e](...r), d = n ? bt : t ? Wt : Ut;
		return !t && N(a, "iterate", l ? Qe : Ze), s(Object.create(u), { next() {
			let { value: e, done: t } = u.next();
			return t ? {
				value: e,
				done: t
			} : {
				value: c ? [d(e[0]), d(e[1])] : d(e),
				done: t
			};
		} });
	};
}
function Ct(e) {
	return function(...t) {
		return e === "delete" ? !1 : e === "clear" ? void 0 : this;
	};
}
function wt(e, t) {
	let n = {
		get(n) {
			let r = this.__v_raw, i = /* @__PURE__ */ P(r), a = /* @__PURE__ */ P(n);
			e || (D(n, a) && N(i, "get", n), N(i, "get", a));
			let { has: o } = xt(i), s = t ? bt : e ? Wt : Ut;
			if (o.call(i, n)) return s(r.get(n));
			if (o.call(i, a)) return s(r.get(a));
			r !== i && r.get(n);
		},
		get size() {
			let t = this.__v_raw;
			return !e && N(/* @__PURE__ */ P(t), "iterate", Ze), t.size;
		},
		has(t) {
			let n = this.__v_raw, r = /* @__PURE__ */ P(n), i = /* @__PURE__ */ P(t);
			return e || (D(t, i) && N(r, "has", t), N(r, "has", i)), t === i ? n.has(t) : n.has(t) || n.has(i);
		},
		forEach(n, r) {
			let i = this, a = i.__v_raw, o = /* @__PURE__ */ P(a), s = t ? bt : e ? Wt : Ut;
			return !e && N(o, "iterate", Ze), a.forEach((e, t) => n.call(r, s(e), s(t), i));
		}
	};
	return s(n, e ? {
		add: Ct("add"),
		set: Ct("set"),
		delete: Ct("delete"),
		clear: Ct("clear")
	} : {
		add(e) {
			let n = /* @__PURE__ */ P(this), r = xt(n), i = /* @__PURE__ */ P(e), a = !t && !/* @__PURE__ */ Bt(e) && !/* @__PURE__ */ zt(e) ? i : e;
			return r.has.call(n, a) || D(e, a) && r.has.call(n, e) || D(i, a) && r.has.call(n, i) || (n.add(a), et(n, "add", a, a)), this;
		},
		set(e, n) {
			!t && !/* @__PURE__ */ Bt(n) && !/* @__PURE__ */ zt(n) && (n = /* @__PURE__ */ P(n));
			let r = /* @__PURE__ */ P(this), { has: i, get: a } = xt(r), o = i.call(r, e);
			o ||= (e = /* @__PURE__ */ P(e), i.call(r, e));
			let s = a.call(r, e);
			return r.set(e, n), o ? D(n, s) && et(r, "set", e, n, s) : et(r, "add", e, n), this;
		},
		delete(e) {
			let t = /* @__PURE__ */ P(this), { has: n, get: r } = xt(t), i = n.call(t, e);
			i ||= (e = /* @__PURE__ */ P(e), n.call(t, e));
			let a = r ? r.call(t, e) : void 0, o = t.delete(e);
			return i && et(t, "delete", e, void 0, a), o;
		},
		clear() {
			let e = /* @__PURE__ */ P(this), t = e.size !== 0, n = e.clear();
			return t && et(e, "clear", void 0, void 0, void 0), n;
		}
	}), [
		"keys",
		"values",
		"entries",
		Symbol.iterator
	].forEach((r) => {
		n[r] = St(r, e, t);
	}), n;
}
function Tt(e, t) {
	let n = wt(e, t);
	return (t, r, i) => r === "__v_isReactive" ? !e : r === "__v_isReadonly" ? e : r === "__v_raw" ? t : Reflect.get(u(n, r) && r in t ? n : t, r, i);
}
var Et = { get: /* @__PURE__ */ Tt(!1, !1) }, Dt = { get: /* @__PURE__ */ Tt(!1, !0) }, Ot = { get: /* @__PURE__ */ Tt(!0, !1) }, kt = /* @__PURE__ */ new WeakMap(), At = /* @__PURE__ */ new WeakMap(), jt = /* @__PURE__ */ new WeakMap(), Mt = /* @__PURE__ */ new WeakMap();
function Nt(e) {
	switch (e) {
		case "Object":
		case "Array": return 1;
		case "Map":
		case "Set":
		case "WeakMap":
		case "WeakSet": return 2;
		default: return 0;
	}
}
// @__NO_SIDE_EFFECTS__
function Pt(e) {
	return /* @__PURE__ */ zt(e) ? e : Lt(e, !1, _t, Et, kt);
}
// @__NO_SIDE_EFFECTS__
function Ft(e) {
	return Lt(e, !1, yt, Dt, At);
}
// @__NO_SIDE_EFFECTS__
function It(e) {
	return Lt(e, !0, vt, Ot, jt);
}
function Lt(e, t, n, r, i) {
	if (!v(e) || e.__v_raw && !(t && e.__v_isReactive) || e.__v_skip || !Object.isExtensible(e)) return e;
	let a = i.get(e);
	if (a) return a;
	let o = Nt(S(e));
	if (o === 0) return e;
	let s = new Proxy(e, o === 2 ? r : n);
	return i.set(e, s), s;
}
// @__NO_SIDE_EFFECTS__
function Rt(e) {
	return /* @__PURE__ */ zt(e) ? /* @__PURE__ */ Rt(e.__v_raw) : !!(e && e.__v_isReactive);
}
// @__NO_SIDE_EFFECTS__
function zt(e) {
	return !!(e && e.__v_isReadonly);
}
// @__NO_SIDE_EFFECTS__
function Bt(e) {
	return !!(e && e.__v_isShallow);
}
// @__NO_SIDE_EFFECTS__
function Vt(e) {
	return e ? !!e.__v_raw : !1;
}
// @__NO_SIDE_EFFECTS__
function P(e) {
	let t = e && e.__v_raw;
	return t ? /* @__PURE__ */ P(t) : e;
}
function Ht(e) {
	return !u(e, "__v_skip") && Object.isExtensible(e) && O(e, "__v_skip", !0), e;
}
var Ut = (e) => v(e) ? /* @__PURE__ */ Pt(e) : e, Wt = (e) => v(e) ? /* @__PURE__ */ It(e) : e;
// @__NO_SIDE_EFFECTS__
function F(e) {
	return e ? e.__v_isRef === !0 : !1;
}
// @__NO_SIDE_EFFECTS__
function Gt(e) {
	return Kt(e, !1);
}
function Kt(e, t) {
	return /* @__PURE__ */ F(e) ? e : new qt(e, t);
}
var qt = class {
	constructor(e, t) {
		this.dep = new Je(), this.__v_isRef = !0, this.__v_isShallow = !1, this._rawValue = t ? e : /* @__PURE__ */ P(e), this._value = t ? e : Ut(e), this.__v_isShallow = t;
	}
	get value() {
		return this.dep.track(), this._value;
	}
	set value(e) {
		let t = this._rawValue, n = this.__v_isShallow || /* @__PURE__ */ Bt(e) || /* @__PURE__ */ zt(e);
		e = n ? e : /* @__PURE__ */ P(e), D(e, t) && (this._rawValue = e, this._value = n ? e : Ut(e), this.dep.trigger());
	}
};
function I(e) {
	return /* @__PURE__ */ F(e) ? e.value : e;
}
var Jt = {
	get: (e, t, n) => t === "__v_raw" ? e : I(Reflect.get(e, t, n)),
	set: (e, t, n, r) => {
		let i = e[t];
		return /* @__PURE__ */ F(i) && !/* @__PURE__ */ F(n) ? (i.value = n, !0) : Reflect.set(e, t, n, r);
	}
};
function Yt(e) {
	return /* @__PURE__ */ Rt(e) ? e : new Proxy(e, Jt);
}
var Xt = class {
	constructor(e, t, n) {
		this.fn = e, this.setter = t, this._value = void 0, this.dep = new Je(this), this.__v_isRef = !0, this.deps = void 0, this.depsTail = void 0, this.flags = 16, this.globalVersion = Ke - 1, this.next = void 0, this.effect = this, this.__v_isReadonly = !t, this.isSSR = n;
	}
	notify() {
		if (this.flags |= 16, !(this.flags & 8) && M !== this) return Me(this, !0), !0;
	}
	get value() {
		let e = this.dep.track();
		return Re(this), e && (e.version = this.dep.version), this._value;
	}
	set value(e) {
		this.setter && this.setter(e);
	}
};
// @__NO_SIDE_EFFECTS__
function Zt(e, t, n = !1) {
	let r, i;
	return h(e) ? r = e : (r = e.get, i = e.set), new Xt(r, i, n);
}
var Qt = {}, $t = /* @__PURE__ */ new WeakMap(), en = void 0;
function tn(e, t = !1, n = en) {
	if (n) {
		let t = $t.get(n);
		t || $t.set(n, t = []), t.push(e);
	}
}
function nn(e, n, i = t) {
	let { immediate: a, deep: o, once: s, scheduler: l, augmentJob: u, call: f } = i, p = (e) => o ? e : /* @__PURE__ */ Bt(e) || o === !1 || o === 0 ? rn(e, 1) : rn(e), m, g, _, v, y = !1, b = !1;
	if (/* @__PURE__ */ F(e) ? (g = () => e.value, y = /* @__PURE__ */ Bt(e)) : /* @__PURE__ */ Rt(e) ? (g = () => p(e), y = !0) : d(e) ? (b = !0, y = e.some((e) => /* @__PURE__ */ Rt(e) || /* @__PURE__ */ Bt(e)), g = () => e.map((e) => {
		if (/* @__PURE__ */ F(e)) return e.value;
		if (/* @__PURE__ */ Rt(e)) return p(e);
		if (h(e)) return f ? f(e, 2) : e();
	})) : g = h(e) ? n ? f ? () => f(e, 2) : e : () => {
		if (_) {
			Ue();
			try {
				_();
			} finally {
				We();
			}
		}
		let t = en;
		en = m;
		try {
			return f ? f(e, 3, [v]) : e(v);
		} finally {
			en = t;
		}
	} : r, n && o) {
		let e = g, t = o === !0 ? Infinity : o;
		g = () => rn(e(), t);
	}
	let x = Ee(), S = () => {
		m.stop(), x && x.active && c(x.effects, m);
	};
	if (s && n) {
		let e = n;
		n = (...t) => {
			let n = e(...t);
			return S(), n;
		};
	}
	let C = b ? Array(e.length).fill(Qt) : Qt, w = (e) => {
		if (!(!(m.flags & 1) || !m.dirty && !e)) if (n) {
			let t = m.run();
			if (e || o || y || (b ? t.some((e, t) => D(e, C[t])) : D(t, C))) {
				_ && _();
				let e = en;
				en = m;
				try {
					let e = [
						t,
						C === Qt ? void 0 : b && C[0] === Qt ? [] : C,
						v
					];
					C = t, f ? f(n, 3, e) : n(...e);
				} finally {
					en = e;
				}
			}
		} else m.run();
	};
	return u && u(w), m = new Oe(g), m.scheduler = l ? () => l(w, !1) : w, v = (e) => tn(e, !1, m), _ = m.onStop = () => {
		let e = $t.get(m);
		if (e) {
			if (f) f(e, 4);
			else for (let t of e) t();
			$t.delete(m);
		}
	}, n ? a ? w(!0) : C = m.run() : l ? l(w.bind(null, !0), !0) : m.run(), S.pause = m.pause.bind(m), S.resume = m.resume.bind(m), S.stop = S, S;
}
function rn(e, t = Infinity, n) {
	if (t <= 0 || !v(e) || e.__v_skip || (n ||= /* @__PURE__ */ new Map(), (n.get(e) || 0) >= t)) return e;
	if (n.set(e, t), t--, /* @__PURE__ */ F(e)) rn(e.value, t, n);
	else if (d(e)) for (let r = 0; r < e.length; r++) rn(e[r], t, n);
	else if (p(e) || f(e)) e.forEach((e) => {
		rn(e, t, n);
	});
	else if (C(e)) {
		for (let r in e) rn(e[r], t, n);
		for (let r of Object.getOwnPropertySymbols(e)) Object.prototype.propertyIsEnumerable.call(e, r) && rn(e[r], t, n);
	}
	return e;
}
//#endregion
//#region node_modules/@vue/runtime-core/dist/runtime-core.esm-bundler.js
function an(e, t, n, r) {
	try {
		return r ? e(...r) : e();
	} catch (e) {
		sn(e, t, n);
	}
}
function on(e, t, n, r) {
	if (h(e)) {
		let i = an(e, t, n, r);
		return i && y(i) && i.catch((e) => {
			sn(e, t, n);
		}), i;
	}
	if (d(e)) {
		let i = [];
		for (let a = 0; a < e.length; a++) i.push(on(e[a], t, n, r));
		return i;
	}
}
function sn(e, n, r, i = !0) {
	let a = n ? n.vnode : null, { errorHandler: o, throwUnhandledErrorInProduction: s } = n && n.appContext.config || t;
	if (n) {
		let t = n.parent, i = n.proxy, a = `https://vuejs.org/error-reference/#runtime-${r}`;
		for (; t;) {
			let n = t.ec;
			if (n) {
				for (let t = 0; t < n.length; t++) if (n[t](e, i, a) === !1) return;
			}
			t = t.parent;
		}
		if (o) {
			Ue(), an(o, null, 10, [
				e,
				i,
				a
			]), We();
			return;
		}
	}
	cn(e, r, a, i, s);
}
function cn(e, t, n, r = !0, i = !1) {
	if (i) throw e;
	console.error(e);
}
var L = [], ln = -1, un = [], dn = null, fn = 0, pn = /* @__PURE__ */ Promise.resolve(), mn = null;
function hn(e) {
	let t = mn || pn;
	return e ? t.then(this ? e.bind(this) : e) : t;
}
function gn(e) {
	let t = ln + 1, n = L.length;
	for (; t < n;) {
		let r = t + n >>> 1, i = L[r], a = Sn(i);
		a < e || a === e && i.flags & 2 ? t = r + 1 : n = r;
	}
	return t;
}
function _n(e) {
	if (!(e.flags & 1)) {
		let t = Sn(e), n = L[L.length - 1];
		!n || !(e.flags & 2) && t >= Sn(n) ? L.push(e) : L.splice(gn(t), 0, e), e.flags |= 1, vn();
	}
}
function vn() {
	mn ||= pn.then(Cn);
}
function yn(e) {
	d(e) ? un.push(...e) : dn && e.id === -1 ? dn.splice(fn + 1, 0, e) : e.flags & 1 || (un.push(e), e.flags |= 1), vn();
}
function bn(e, t, n = ln + 1) {
	for (; n < L.length; n++) {
		let t = L[n];
		if (t && t.flags & 2) {
			if (e && t.id !== e.uid) continue;
			L.splice(n, 1), n--, t.flags & 4 && (t.flags &= -2), t(), t.flags & 4 || (t.flags &= -2);
		}
	}
}
function xn(e) {
	if (un.length) {
		let e = [...new Set(un)].sort((e, t) => Sn(e) - Sn(t));
		if (un.length = 0, dn) {
			dn.push(...e);
			return;
		}
		for (dn = e, fn = 0; fn < dn.length; fn++) {
			let e = dn[fn];
			e.flags & 4 && (e.flags &= -2), e.flags & 8 || e(), e.flags &= -2;
		}
		dn = null, fn = 0;
	}
}
var Sn = (e) => e.id == null ? e.flags & 2 ? -1 : Infinity : e.id;
function Cn(e) {
	try {
		for (ln = 0; ln < L.length; ln++) {
			let e = L[ln];
			e && !(e.flags & 8) && (e.flags & 4 && (e.flags &= -2), an(e, e.i, e.i ? 15 : 14), e.flags & 4 || (e.flags &= -2));
		}
	} finally {
		for (; ln < L.length; ln++) {
			let e = L[ln];
			e && (e.flags &= -2);
		}
		ln = -1, L.length = 0, xn(e), mn = null, (L.length || un.length) && Cn(e);
	}
}
var wn = null, Tn = null;
function En(e) {
	let t = wn;
	return wn = e, Tn = e && e.type.__scopeId || null, t;
}
function Dn(e, t = wn, n) {
	if (!t || e._n) return e;
	let r = (...n) => {
		r._d && ea(-1);
		let i = En(t), a = Xi.length, o;
		try {
			o = e(...n);
		} finally {
			for (let e = Xi.length; e > a; e--) Qi();
			En(i), r._d && ea(1);
		}
		return o;
	};
	return r._n = !0, r._c = !0, r._d = !0, r;
}
function R(e, n) {
	if (wn === null) return e;
	let r = Ia(wn), i = e.dirs ||= [];
	for (let e = 0; e < n.length; e++) {
		let [a, o, s, c = t] = n[e];
		a && (h(a) && (a = {
			mounted: a,
			updated: a
		}), a.deep && rn(o), i.push({
			dir: a,
			instance: r,
			value: o,
			oldValue: void 0,
			arg: s,
			modifiers: c
		}));
	}
	return e;
}
function On(e, t, n, r) {
	let i = e.dirs, a = t && t.dirs;
	for (let o = 0; o < i.length; o++) {
		let s = i[o];
		a && (s.oldValue = a[o].value);
		let c = s.dir[r];
		c && (Ue(), on(c, n, 8, [
			e.el,
			s,
			e,
			t
		]), We());
	}
}
function kn(e, t) {
	if (ba) {
		let n = ba.provides, r = ba.parent && ba.parent.provides;
		r === n && (n = ba.provides = Object.create(r)), n[e] = t;
	}
}
function An(e, t, n = !1) {
	let r = xa();
	if (r || ai) {
		let i = ai ? ai._context.provides : r ? r.parent == null || r.ce ? r.vnode.appContext && r.vnode.appContext.provides : r.parent.provides : void 0;
		if (i && e in i) return i[e];
		if (arguments.length > 1) return n && h(t) ? t.call(r && r.proxy) : t;
	}
}
var jn = /* @__PURE__ */ Symbol.for("v-scx"), Mn = () => An(jn);
function Nn(e, t, n) {
	return Pn(e, t, n);
}
function Pn(e, n, i = t) {
	let { immediate: a, deep: o, flush: c, once: l } = i, u = s({}, i), d = n && a || !n && c !== "post", f;
	if (Da) {
		if (c === "sync") {
			let e = Mn();
			f = e.__watcherHandles ||= [];
		} else if (!d) {
			let e = () => {};
			return e.stop = r, e.resume = r, e.pause = r, e;
		}
	}
	let p = ba;
	u.call = (e, t, n) => on(e, p, t, n);
	let m = !1;
	c === "post" ? u.scheduler = (e) => {
		V(e, p && p.suspense);
	} : c !== "sync" && (m = !0, u.scheduler = (e, t) => {
		t ? e() : _n(e);
	}), u.augmentJob = (e) => {
		n && (e.flags |= 4), m && (e.flags |= 2, p && (e.id = p.uid, e.i = p));
	};
	let h = nn(e, n, u);
	return Da && (f ? f.push(h) : d && h()), h;
}
function Fn(e, t, n) {
	let r = this.proxy, i = g(e) ? e.includes(".") ? In(r, e) : () => r[e] : e.bind(r, r), a;
	h(t) ? a = t : (a = t.handler, n = t);
	let o = wa(this), s = Pn(i, a.bind(r), n);
	return o(), s;
}
function In(e, t) {
	let n = t.split(".");
	return () => {
		let t = e;
		for (let e = 0; e < n.length && t; e++) t = t[n[e]];
		return t;
	};
}
var Ln = /* @__PURE__ */ new WeakMap(), Rn = /* @__PURE__ */ Symbol("_vte"), zn = (e) => e.__isTeleport, Bn = (e) => e && (e.disabled || e.disabled === ""), Vn = (e) => e && (e.defer || e.defer === ""), Hn = (e) => typeof SVGElement < "u" && e instanceof SVGElement, Un = (e) => typeof MathMLElement == "function" && e instanceof MathMLElement, Wn = (e, t) => {
	let n = e && e.to;
	return g(n) ? t ? t(n) : null : n;
}, Gn = {
	name: "Teleport",
	__isTeleport: !0,
	process(e, t, n, r, i, a, o, s, c, l) {
		let { mc: u, pc: d, pbc: f, o: { insert: p, querySelector: m, createText: h, createComment: g, parentNode: _ } } = l, v = Bn(t.props), { dynamicChildren: y } = t, b = (e, t, n) => {
			e.shapeFlag & 16 && u(e.children, t, n, i, a, o, s, c);
		}, x = (e = t) => {
			let n = Bn(e.props), r = e.target = Wn(e.props, m), a = Xn(r, e, h, p);
			r && (o !== "svg" && Hn(r) ? o = "svg" : o !== "mathml" && Un(r) && (o = "mathml"), i && i.isCE && (i.ce._teleportTargets || (i.ce._teleportTargets = /* @__PURE__ */ new Set())).add(r), n || (b(e, r, a), Yn(e, !1)));
		}, S = (e) => {
			let t = () => {
				if (Ln.get(e) === t) {
					if (Ln.delete(e), Bn(e.props)) {
						let t = _(e.el) || n;
						b(e, t, e.anchor), Yn(e, !0);
					}
					x(e);
				}
			};
			Ln.set(e, t), V(t, a);
		};
		if (e == null) {
			let e = t.el = h(""), i = t.anchor = h("");
			if (p(e, n, r), p(i, n, r), Vn(t.props) || a && a.pendingBranch) {
				S(t);
				return;
			}
			v && (b(t, n, i), Yn(t, !0)), x();
		} else {
			t.el = e.el;
			let r = t.anchor = e.anchor, u = Ln.get(e);
			if (u) {
				u.flags |= 8, Ln.delete(e), S(t);
				return;
			}
			t.targetStart = e.targetStart;
			let p = t.target = e.target, h = t.targetAnchor = e.targetAnchor, g = Bn(e.props), _ = g ? n : p, b = g ? r : h;
			if (o === "svg" || Hn(p) ? o = "svg" : (o === "mathml" || Un(p)) && (o = "mathml"), y ? (f(e.dynamicChildren, y, _, i, a, o, s), Vi(e, t, !0)) : c || d(e, t, _, b, i, a, o, s, !1), v) g ? t.props && e.props && t.props.to !== e.props.to && (t.props.to = e.props.to) : Kn(t, n, r, l, 1);
			else if ((t.props && t.props.to) !== (e.props && e.props.to)) {
				let e = Wn(t.props, m);
				e && (t.target = e, Kn(t, e, null, l, 0));
			} else g && Kn(t, p, h, l, 1);
			Yn(t, v);
		}
	},
	remove(e, t, n, { um: r, o: { remove: i } }, a) {
		let { shapeFlag: o, children: s, anchor: c, targetStart: l, targetAnchor: u, target: d, props: f } = e, p = Bn(f), m = a || !p, h = Ln.get(e);
		if (h && (h.flags |= 8, Ln.delete(e)), d && (i(l), i(u)), a && i(c), !h && (p || d) && o & 16) for (let e = 0; e < s.length; e++) {
			let i = s[e];
			r(i, t, n, m, !!i.dynamicChildren);
		}
	},
	move: Kn,
	hydrate: qn
};
function Kn(e, t, n, { o: { insert: r }, m: i }, a = 2) {
	a === 0 && r(e.targetAnchor, t, n);
	let { el: o, anchor: s, shapeFlag: c, children: l, props: u } = e, d = a === 2;
	if (d && r(o, t, n), !Ln.has(e) && (!d || Bn(u)) && c & 16) for (let e = 0; e < l.length; e++) i(l[e], t, n, 2);
	d && r(s, t, n);
}
function qn(e, t, n, r, i, a, { o: { nextSibling: o, parentNode: s, querySelector: c, insert: l, createText: u } }, d) {
	function f(e, n) {
		let r = n;
		for (; r;) {
			if (r && r.nodeType === 8) {
				if (r.data === "teleport start anchor") t.targetStart = r;
				else if (r.data === "teleport anchor") {
					t.targetAnchor = r, e._lpa = t.targetAnchor && o(t.targetAnchor);
					break;
				}
			}
			r = o(r);
		}
	}
	function p(e, t) {
		t.anchor = d(o(e), t, s(e), n, r, i, a);
	}
	let m = t.target = Wn(t.props, c), h = Bn(t.props);
	if (m) {
		let c = m._lpa || m.firstChild;
		t.shapeFlag & 16 && (h ? (p(e, t), f(m, c), t.targetAnchor || Xn(m, t, u, l, s(e) === m ? e : null)) : (t.anchor = o(e), f(m, c), t.targetAnchor || Xn(m, t, u, l), d(c && o(c), t, m, n, r, i, a))), Yn(t, h);
	} else h && t.shapeFlag & 16 && (p(e, t), t.targetStart = e, t.targetAnchor = o(e));
	return t.anchor && o(t.anchor);
}
var Jn = Gn;
function Yn(e, t) {
	let n = e.ctx;
	if (n && n.ut) {
		let r, i;
		for (t ? (r = e.el, i = e.anchor) : (r = e.targetStart, i = e.targetAnchor); r && r !== i;) r.nodeType === 1 && r.setAttribute("data-v-owner", n.uid), r = r.nextSibling;
		n.ut();
	}
}
function Xn(e, t, n, r, i = null) {
	let a = t.targetStart = n(""), o = t.targetAnchor = n("");
	return a[Rn] = o, e && (r(a, e, i), r(o, e, i)), o;
}
var Zn = /* @__PURE__ */ Symbol("_leaveCb"), Qn = /* @__PURE__ */ Symbol("_enterCb");
function $n() {
	let e = {
		isMounted: !1,
		isLeaving: !1,
		isUnmounting: !1,
		leavingVNodes: /* @__PURE__ */ new Map()
	};
	return Dr(() => {
		e.isMounted = !0;
	}), Ar(() => {
		e.isUnmounting = !0;
	}), e;
}
var er = [Function, Array], tr = {
	mode: String,
	appear: Boolean,
	persisted: Boolean,
	onBeforeEnter: er,
	onEnter: er,
	onAfterEnter: er,
	onEnterCancelled: er,
	onBeforeLeave: er,
	onLeave: er,
	onAfterLeave: er,
	onLeaveCancelled: er,
	onBeforeAppear: er,
	onAppear: er,
	onAfterAppear: er,
	onAppearCancelled: er
}, nr = (e) => {
	let t = e.subTree;
	return t.component ? nr(t.component) : t;
}, rr = {
	name: "BaseTransition",
	props: tr,
	setup(e, { slots: t }) {
		let n = xa(), r = $n();
		return () => {
			let i = t.default && dr(t.default(), !0), a = i && i.length ? ir(i) : n.subTree ? J() : void 0;
			if (!a) return;
			let o = /* @__PURE__ */ P(e), { mode: s } = o;
			if (r.isLeaving) return cr(a);
			let c = lr(a);
			if (!c) return cr(a);
			let l = sr(c, o, r, n, (e) => l = e);
			c.type !== U && ur(c, l);
			let u = n.subTree && lr(n.subTree);
			if (u && u.type !== U && !ia(u, c) && nr(n).type !== U) {
				let e = sr(u, o, r, n);
				if (ur(u, e), s === "out-in" && c.type !== U) return r.isLeaving = !0, e.afterLeave = () => {
					r.isLeaving = !1, n.job.flags & 8 || n.update(), delete e.afterLeave, u = void 0;
				}, cr(a);
				s === "in-out" && c.type !== U ? e.delayLeave = (e, t, n) => {
					let i = or(r, u);
					i[String(u.key)] = u, e[Zn] = () => {
						t(), e[Zn] = void 0, delete l.delayedLeave, u = void 0;
					}, l.delayedLeave = () => {
						n(), delete l.delayedLeave, u = void 0;
					};
				} : u = void 0;
			} else u &&= void 0;
			return a;
		};
	}
};
function ir(e) {
	let t = e[0];
	if (e.length > 1) {
		for (let n of e) if (n.type !== U) {
			t = n;
			break;
		}
	}
	return t;
}
var ar = rr;
function or(e, t) {
	let { leavingVNodes: n } = e, r = n.get(t.type);
	return r || (r = /* @__PURE__ */ Object.create(null), n.set(t.type, r)), r;
}
function sr(e, t, n, r, i) {
	let { appear: a, mode: o, persisted: s = !1, onBeforeEnter: c, onEnter: l, onAfterEnter: u, onEnterCancelled: f, onBeforeLeave: p, onLeave: m, onAfterLeave: h, onLeaveCancelled: g, onBeforeAppear: _, onAppear: v, onAfterAppear: y, onAppearCancelled: b } = t, x = String(e.key), S = or(n, e), C = (e, t) => {
		e && on(e, r, 9, t);
	}, w = (e, t) => {
		let n = t[1];
		C(e, t), d(e) ? e.every((e) => e.length <= 1) && n() : e.length <= 1 && n();
	}, ee = {
		mode: o,
		persisted: s,
		beforeEnter(t) {
			let r = c;
			if (!n.isMounted) if (a) r = _ || c;
			else return;
			t[Zn] && t[Zn](!0);
			let i = S[x];
			i && ia(e, i) && i.el[Zn] && i.el[Zn](), C(r, [t]);
		},
		enter(t) {
			if (S[x] === e) return;
			let r = l, i = u, o = f;
			if (!n.isMounted) if (a) r = v || l, i = y || u, o = b || f;
			else return;
			let s = !1;
			t[Qn] = (e) => {
				s || (s = !0, C(e ? o : i, [t]), ee.delayedLeave && ee.delayedLeave(), t[Qn] = void 0);
			};
			let c = t[Qn].bind(null, !1);
			r ? w(r, [t, c]) : c();
		},
		leave(t, r) {
			let i = String(e.key);
			if (t[Qn] && t[Qn](!0), n.isUnmounting) return r();
			C(p, [t]);
			let a = !1;
			t[Zn] = (n) => {
				a || (a = !0, r(), C(n ? g : h, [t]), t[Zn] = void 0, S[i] === e && delete S[i]);
			};
			let o = t[Zn].bind(null, !1);
			S[i] = e, m ? w(m, [t, o]) : o();
		},
		clone(e) {
			let a = sr(e, t, n, r, i);
			return i && i(a), a;
		}
	};
	return ee;
}
function cr(e) {
	if (yr(e)) return e = la(e), e.children = null, e;
}
function lr(e) {
	if (!yr(e)) return zn(e.type) && e.children ? ir(e.children) : e;
	if (e.component) return e.component.subTree;
	let { shapeFlag: t, children: n } = e;
	if (n) {
		if (t & 16) return n[0];
		if (t & 32 && h(n.default)) return n.default();
	}
}
function ur(e, t) {
	e.shapeFlag & 6 && e.component ? (e.transition = t, ur(e.component.subTree, t)) : e.shapeFlag & 128 ? (e.ssContent.transition = t.clone(e.ssContent), e.ssFallback.transition = t.clone(e.ssFallback)) : e.transition = t;
}
function dr(e, t = !1, n) {
	let r = [], i = 0;
	for (let a = 0; a < e.length; a++) {
		let o = e[a], s = n == null ? o.key : String(n) + String(o.key == null ? a : o.key);
		o.type === H ? (o.patchFlag & 128 && i++, r = r.concat(dr(o.children, t, s))) : (t || o.type !== U) && r.push(s == null ? o : la(o, { key: s }));
	}
	if (i > 1) for (let e = 0; e < r.length; e++) r[e].patchFlag = -2;
	return r;
}
// @__NO_SIDE_EFFECTS__
function fr(e, t) {
	return h(e) ? /* @__PURE__ */ s({ name: e.name }, t, { setup: e }) : e;
}
function pr(e) {
	e.ids = [
		e.ids[0] + e.ids[2]++ + "-",
		0,
		0
	];
}
function mr(e, t) {
	let n;
	return !!((n = Object.getOwnPropertyDescriptor(e, t)) && !n.configurable);
}
var hr = /* @__PURE__ */ new WeakMap();
function gr(e, n, r, a, o = !1) {
	if (d(e)) {
		e.forEach((e, t) => gr(e, n && (d(n) ? n[t] : n), r, a, o));
		return;
	}
	if (vr(a) && !o) {
		a.shapeFlag & 512 && a.type.__asyncResolved && a.component.subTree.component && gr(e, n, r, a.component.subTree);
		return;
	}
	let s = a.shapeFlag & 4 ? Ia(a.component) : a.el, l = o ? null : s, { i: f, r: p } = e, m = n && n.r, _ = f.refs === t ? f.refs = {} : f.refs, v = f.setupState, y = /* @__PURE__ */ P(v), b = v === t ? i : (e) => !mr(_, e) && u(y, e), x = (e, t) => !(t && mr(_, t));
	if (m != null && m !== p) {
		if (_r(n), g(m)) _[m] = null, b(m) && (v[m] = null);
		else if (/* @__PURE__ */ F(m)) {
			let e = n;
			x(m, e.k) && (m.value = null), e.k && (_[e.k] = null);
		}
	}
	if (h(p)) an(p, f, 12, [l, _]);
	else {
		let t = g(p), n = /* @__PURE__ */ F(p);
		if (t || n) {
			let i = () => {
				if (e.f) {
					let n = t ? b(p) ? v[p] : _[p] : x(p) || !e.k ? p.value : _[e.k];
					if (o) d(n) && c(n, s);
					else if (d(n)) n.includes(s) || n.push(s);
					else if (t) _[p] = [s], b(p) && (v[p] = _[p]);
					else {
						let t = [s];
						x(p, e.k) && (p.value = t), e.k && (_[e.k] = t);
					}
				} else t ? (_[p] = l, b(p) && (v[p] = l)) : n && (x(p, e.k) && (p.value = l), e.k && (_[e.k] = l));
			};
			if (l) {
				let t = () => {
					i(), hr.delete(e);
				};
				t.id = -1, hr.set(e, t), V(t, r);
			} else _r(e), i();
		}
	}
}
function _r(e) {
	let t = hr.get(e);
	t && (t.flags |= 8, hr.delete(e));
}
ue().requestIdleCallback, ue().cancelIdleCallback;
var vr = (e) => !!e.type.__asyncLoader, yr = (e) => e.type.__isKeepAlive;
function br(e, t) {
	Sr(e, "a", t);
}
function xr(e, t) {
	Sr(e, "da", t);
}
function Sr(e, t, n = ba) {
	let r = e.__wdc ||= () => {
		let t = n;
		for (; t;) {
			if (t.isDeactivated) return;
			t = t.parent;
		}
		return e();
	};
	if (wr(t, r, n), n) {
		let e = n.parent;
		for (; e && e.parent;) yr(e.parent.vnode) && Cr(r, t, n, e), e = e.parent;
	}
}
function Cr(e, t, n, r) {
	let i = wr(t, e, r, !0);
	jr(() => {
		c(r[t], i);
	}, n);
}
function wr(e, t, n = ba, r = !1) {
	if (n) {
		let i = n[e] || (n[e] = []), a = t.__weh ||= (...r) => {
			Ue();
			let i = wa(n), a = on(t, n, e, r);
			return i(), We(), a;
		};
		return r ? i.unshift(a) : i.push(a), a;
	}
}
var Tr = (e) => (t, n = ba) => {
	(!Da || e === "sp") && wr(e, (...e) => t(...e), n);
}, Er = Tr("bm"), Dr = Tr("m"), Or = Tr("bu"), kr = Tr("u"), Ar = Tr("bum"), jr = Tr("um"), Mr = Tr("sp"), Nr = Tr("rtg"), Pr = Tr("rtc");
function Fr(e, t = ba) {
	wr("ec", e, t);
}
var Ir = /* @__PURE__ */ Symbol.for("v-ndc");
function z(e, t, n, r) {
	let i, a = n && n[r], o = d(e);
	if (o || g(e)) {
		let n = o && /* @__PURE__ */ Rt(e), r = !1, s = !1;
		n && (r = !/* @__PURE__ */ Bt(e), s = /* @__PURE__ */ zt(e), e = nt(e)), i = Array(e.length);
		for (let n = 0, o = e.length; n < o; n++) i[n] = t(r ? s ? Wt(Ut(e[n])) : Ut(e[n]) : e[n], n, void 0, a && a[n]);
	} else if (typeof e == "number") {
		i = Array(e);
		for (let n = 0; n < e; n++) i[n] = t(n + 1, n, void 0, a && a[n]);
	} else if (v(e)) if (e[Symbol.iterator]) i = Array.from(e, (e, n) => t(e, n, void 0, a && a[n]));
	else {
		let n = Object.keys(e);
		i = Array(n.length);
		for (let r = 0, o = n.length; r < o; r++) {
			let o = n[r];
			i[r] = t(e[o], o, r, a && a[r]);
		}
	}
	else i = [];
	return n && (n[r] = i), i;
}
var Lr = (e) => e ? Ea(e) ? Ia(e) : Lr(e.parent) : null, Rr = /* @__PURE__ */ s(/* @__PURE__ */ Object.create(null), {
	$: (e) => e,
	$el: (e) => e.vnode.el,
	$data: (e) => e.data,
	$props: (e) => e.props,
	$attrs: (e) => e.attrs,
	$slots: (e) => e.slots,
	$refs: (e) => e.refs,
	$parent: (e) => Lr(e.parent),
	$root: (e) => Lr(e.root),
	$host: (e) => e.ce,
	$emit: (e) => e.emit,
	$options: (e) => qr(e),
	$forceUpdate: (e) => e.f ||= () => {
		_n(e.update);
	},
	$nextTick: (e) => e.n ||= hn.bind(e.proxy),
	$watch: (e) => Fn.bind(e)
}), zr = (e, n) => e !== t && !e.__isScriptSetup && u(e, n), Br = {
	get({ _: e }, n) {
		if (n === "__v_skip") return !0;
		let { ctx: r, setupState: i, data: a, props: o, accessCache: s, type: c, appContext: l } = e;
		if (n[0] !== "$") {
			let e = s[n];
			if (e !== void 0) switch (e) {
				case 1: return i[n];
				case 2: return a[n];
				case 4: return r[n];
				case 3: return o[n];
			}
			else if (zr(i, n)) return s[n] = 1, i[n];
			else if (a !== t && u(a, n)) return s[n] = 2, a[n];
			else if (u(o, n)) return s[n] = 3, o[n];
			else if (r !== t && u(r, n)) return s[n] = 4, r[n];
			else Hr && (s[n] = 0);
		}
		let d = Rr[n], f, p;
		if (d) return n === "$attrs" && N(e.attrs, "get", ""), d(e);
		if ((f = c.__cssModules) && (f = f[n])) return f;
		if (r !== t && u(r, n)) return s[n] = 4, r[n];
		if (p = l.config.globalProperties, u(p, n)) return p[n];
	},
	set({ _: e }, n, r) {
		let { data: i, setupState: a, ctx: o } = e;
		return zr(a, n) ? (a[n] = r, !0) : i !== t && u(i, n) ? (i[n] = r, !0) : u(e.props, n) || n[0] === "$" && n.slice(1) in e ? !1 : (o[n] = r, !0);
	},
	has({ _: { data: e, setupState: n, accessCache: r, ctx: i, appContext: a, props: o, type: s } }, c) {
		let l;
		return !!(r[c] || e !== t && c[0] !== "$" && u(e, c) || zr(n, c) || u(o, c) || u(i, c) || u(Rr, c) || u(a.config.globalProperties, c) || (l = s.__cssModules) && l[c]);
	},
	defineProperty(e, t, n) {
		return n.get == null ? u(n, "value") && this.set(e, t, n.value, null) : e._.accessCache[t] = 0, Reflect.defineProperty(e, t, n);
	}
};
function Vr(e) {
	return d(e) ? e.reduce((e, t) => (e[t] = null, e), {}) : e;
}
var Hr = !0;
function Ur(e) {
	let t = qr(e), n = e.proxy, i = e.ctx;
	Hr = !1, t.beforeCreate && Gr(t.beforeCreate, e, "bc");
	let { data: a, computed: o, methods: s, watch: c, provide: l, inject: u, created: f, beforeMount: p, mounted: m, beforeUpdate: g, updated: _, activated: y, deactivated: b, beforeDestroy: x, beforeUnmount: S, destroyed: C, unmounted: w, render: ee, renderTracked: te, renderTriggered: ne, errorCaptured: T, serverPrefetch: re, expose: E, inheritAttrs: ie, components: ae, directives: D, filters: oe } = t;
	if (u && Wr(u, i, null), s) for (let e in s) {
		let t = s[e];
		h(t) && (i[e] = t.bind(n));
	}
	if (a) {
		let t = a.call(n, n);
		v(t) && (e.data = /* @__PURE__ */ Pt(t));
	}
	if (Hr = !0, o) for (let e in o) {
		let t = o[e], a = Y({
			get: h(t) ? t.bind(n, n) : h(t.get) ? t.get.bind(n, n) : r,
			set: !h(t) && h(t.set) ? t.set.bind(n) : r
		});
		Object.defineProperty(i, e, {
			enumerable: !0,
			configurable: !0,
			get: () => a.value,
			set: (e) => a.value = e
		});
	}
	if (c) for (let e in c) Kr(c[e], i, n, e);
	if (l) {
		let e = h(l) ? l.call(n) : l;
		Reflect.ownKeys(e).forEach((t) => {
			kn(t, e[t]);
		});
	}
	f && Gr(f, e, "c");
	function O(e, t) {
		d(t) ? t.forEach((t) => e(t.bind(n))) : t && e(t.bind(n));
	}
	if (O(Er, p), O(Dr, m), O(Or, g), O(kr, _), O(br, y), O(xr, b), O(Fr, T), O(Pr, te), O(Nr, ne), O(Ar, S), O(jr, w), O(Mr, re), d(E)) if (E.length) {
		let t = e.exposed ||= {};
		E.forEach((e) => {
			Object.defineProperty(t, e, {
				get: () => n[e],
				set: (t) => n[e] = t,
				enumerable: !0
			});
		});
	} else e.exposed ||= {};
	ee && e.render === r && (e.render = ee), ie != null && (e.inheritAttrs = ie), ae && (e.components = ae), D && (e.directives = D), re && pr(e);
}
function Wr(e, t, n = r) {
	d(e) && (e = Qr(e));
	for (let n in e) {
		let r = e[n], i;
		i = v(r) ? "default" in r ? An(r.from || n, r.default, !0) : An(r.from || n) : An(r), /* @__PURE__ */ F(i) ? Object.defineProperty(t, n, {
			enumerable: !0,
			configurable: !0,
			get: () => i.value,
			set: (e) => i.value = e
		}) : t[n] = i;
	}
}
function Gr(e, t, n) {
	on(d(e) ? e.map((e) => e.bind(t.proxy)) : e.bind(t.proxy), t, n);
}
function Kr(e, t, n, r) {
	let i = r.includes(".") ? In(n, r) : () => n[r];
	if (g(e)) {
		let n = t[e];
		h(n) && Nn(i, n);
	} else if (h(e)) Nn(i, e.bind(n));
	else if (v(e)) if (d(e)) e.forEach((e) => Kr(e, t, n, r));
	else {
		let r = h(e.handler) ? e.handler.bind(n) : t[e.handler];
		h(r) && Nn(i, r, e);
	}
}
function qr(e) {
	let t = e.type, { mixins: n, extends: r } = t, { mixins: i, optionsCache: a, config: { optionMergeStrategies: o } } = e.appContext, s = a.get(t), c;
	return s ? c = s : !i.length && !n && !r ? c = t : (c = {}, i.length && i.forEach((e) => Jr(c, e, o, !0)), Jr(c, t, o)), v(t) && a.set(t, c), c;
}
function Jr(e, t, n, r = !1) {
	let { mixins: i, extends: a } = t;
	a && Jr(e, a, n, !0), i && i.forEach((t) => Jr(e, t, n, !0));
	for (let i in t) if (!(r && i === "expose")) {
		let r = Yr[i] || n && n[i];
		e[i] = r ? r(e[i], t[i]) : t[i];
	}
	return e;
}
var Yr = {
	data: Xr,
	props: ei,
	emits: ei,
	methods: $r,
	computed: $r,
	beforeCreate: B,
	created: B,
	beforeMount: B,
	mounted: B,
	beforeUpdate: B,
	updated: B,
	beforeDestroy: B,
	beforeUnmount: B,
	destroyed: B,
	unmounted: B,
	activated: B,
	deactivated: B,
	errorCaptured: B,
	serverPrefetch: B,
	components: $r,
	directives: $r,
	watch: ti,
	provide: Xr,
	inject: Zr
};
function Xr(e, t) {
	return t ? e ? function() {
		return s(h(e) ? e.call(this, this) : e, h(t) ? t.call(this, this) : t);
	} : t : e;
}
function Zr(e, t) {
	return $r(Qr(e), Qr(t));
}
function Qr(e) {
	if (d(e)) {
		let t = {};
		for (let n = 0; n < e.length; n++) t[e[n]] = e[n];
		return t;
	}
	return e;
}
function B(e, t) {
	return e ? [...new Set([].concat(e, t))] : t;
}
function $r(e, t) {
	return e ? s(/* @__PURE__ */ Object.create(null), e, t) : t;
}
function ei(e, t) {
	return e ? d(e) && d(t) ? [.../* @__PURE__ */ new Set([...e, ...t])] : s(/* @__PURE__ */ Object.create(null), Vr(e), Vr(t ?? {})) : t;
}
function ti(e, t) {
	if (!e) return t;
	if (!t) return e;
	let n = s(/* @__PURE__ */ Object.create(null), e);
	for (let r in t) n[r] = B(e[r], t[r]);
	return n;
}
function ni() {
	return {
		app: null,
		config: {
			isNativeTag: i,
			performance: !1,
			globalProperties: {},
			optionMergeStrategies: {},
			errorHandler: void 0,
			warnHandler: void 0,
			compilerOptions: {}
		},
		mixins: [],
		components: {},
		directives: {},
		provides: /* @__PURE__ */ Object.create(null),
		optionsCache: /* @__PURE__ */ new WeakMap(),
		propsCache: /* @__PURE__ */ new WeakMap(),
		emitsCache: /* @__PURE__ */ new WeakMap()
	};
}
var ri = 0;
function ii(e, t) {
	return function(n, r = null) {
		h(n) || (n = s({}, n)), r != null && !v(r) && (r = null);
		let i = ni(), a = /* @__PURE__ */ new WeakSet(), o = [], c = !1, l = i.app = {
			_uid: ri++,
			_component: n,
			_props: r,
			_container: null,
			_context: i,
			_instance: null,
			version: za,
			get config() {
				return i.config;
			},
			set config(e) {},
			use(e, ...t) {
				return a.has(e) || (e && h(e.install) ? (a.add(e), e.install(l, ...t)) : h(e) && (a.add(e), e(l, ...t))), l;
			},
			mixin(e) {
				return i.mixins.includes(e) || i.mixins.push(e), l;
			},
			component(e, t) {
				return t ? (i.components[e] = t, l) : i.components[e];
			},
			directive(e, t) {
				return t ? (i.directives[e] = t, l) : i.directives[e];
			},
			mount(a, o, s) {
				if (!c) {
					let u = l._ceVNode || q(n, r);
					return u.appContext = i, s === !0 ? s = "svg" : s === !1 && (s = void 0), o && t ? t(u, a) : e(u, a, s), c = !0, l._container = a, a.__vue_app__ = l, Ia(u.component);
				}
			},
			onUnmount(e) {
				o.push(e);
			},
			unmount() {
				c && (on(o, l._instance, 16), e(null, l._container), delete l._container.__vue_app__);
			},
			provide(e, t) {
				return i.provides[e] = t, l;
			},
			runWithContext(e) {
				let t = ai;
				ai = l;
				try {
					return e();
				} finally {
					ai = t;
				}
			}
		};
		return l;
	};
}
var ai = null, oi = (e, t) => t === "modelValue" || t === "model-value" ? e.modelModifiers : e[`${t}Modifiers`] || e[`${T(t)}Modifiers`] || e[`${E(t)}Modifiers`];
function si(e, n, ...r) {
	if (e.isUnmounted) return;
	let i = e.vnode.props || t, a = r, o = n.startsWith("update:"), s = o && oi(i, n.slice(7));
	s && (s.trim && (a = r.map((e) => g(e) ? e.trim() : e)), s.number && (a = r.map(se)));
	let c, l = i[c = ae(n)] || i[c = ae(T(n))];
	!l && o && (l = i[c = ae(E(n))]), l && on(l, e, 6, a);
	let u = i[c + "Once"];
	if (u) {
		if (!e.emitted) e.emitted = {};
		else if (e.emitted[c]) return;
		e.emitted[c] = !0, on(u, e, 6, a);
	}
}
var ci = /* @__PURE__ */ new WeakMap();
function li(e, t, n = !1) {
	let r = n ? ci : t.emitsCache, i = r.get(e);
	if (i !== void 0) return i;
	let a = e.emits, o = {}, c = !1;
	if (!h(e)) {
		let r = (e) => {
			let n = li(e, t, !0);
			n && (c = !0, s(o, n));
		};
		!n && t.mixins.length && t.mixins.forEach(r), e.extends && r(e.extends), e.mixins && e.mixins.forEach(r);
	}
	return !a && !c ? (v(e) && r.set(e, null), null) : (d(a) ? a.forEach((e) => o[e] = null) : s(o, a), v(e) && r.set(e, o), o);
}
function ui(e, t) {
	return !e || !a(t) ? !1 : (t = t.slice(2), t = t === "Once" ? t : t.replace(/Once$/, ""), u(e, t[0].toLowerCase() + t.slice(1)) || u(e, E(t)) || u(e, t));
}
function di(e) {
	let { type: t, vnode: n, proxy: r, withProxy: i, propsOptions: [a], slots: s, attrs: c, emit: l, render: u, renderCache: d, props: f, data: p, setupState: m, ctx: h, inheritAttrs: g } = e, _ = En(e), v, y;
	try {
		if (n.shapeFlag & 4) {
			let e = i || r, t = e;
			v = fa(u.call(t, e, d, f, m, p, h)), y = c;
		} else {
			let e = t;
			v = fa(e.length > 1 ? e(f, {
				attrs: c,
				slots: s,
				emit: l
			}) : e(f, null)), y = t.props ? c : fi(c);
		}
	} catch (t) {
		Xi.length = 0, sn(t, e, 1), v = q(U);
	}
	let b = v;
	if (y && g !== !1) {
		let e = Object.keys(y), { shapeFlag: t } = b;
		e.length && t & 7 && (a && e.some(o) && (y = pi(y, a)), b = la(b, y, !1, !0));
	}
	return n.dirs && (b = la(b, null, !1, !0), b.dirs = b.dirs ? b.dirs.concat(n.dirs) : n.dirs), n.transition && ur(b, n.transition), v = b, En(_), v;
}
var fi = (e) => {
	let t;
	for (let n in e) (n === "class" || n === "style" || a(n)) && ((t ||= {})[n] = e[n]);
	return t;
}, pi = (e, t) => {
	let n = {};
	for (let r in e) (!o(r) || !(r.slice(9) in t)) && (n[r] = e[r]);
	return n;
};
function mi(e, t, n) {
	let { props: r, children: i, component: a } = e, { props: o, children: s, patchFlag: c } = t, l = a.emitsOptions;
	if (t.dirs || t.transition) return !0;
	if (n && c >= 0) {
		if (c & 1024) return !0;
		if (c & 16) return r ? hi(r, o, l) : !!o;
		if (c & 8) {
			let e = t.dynamicProps;
			for (let t = 0; t < e.length; t++) {
				let n = e[t];
				if (gi(o, r, n) && !ui(l, n)) return !0;
			}
		}
	} else return (i || s) && (!s || !s.$stable) ? !0 : r === o ? !1 : r ? !o || hi(r, o, l) : !!o;
	return !1;
}
function hi(e, t, n) {
	let r = Object.keys(t);
	if (r.length !== Object.keys(e).length) return !0;
	for (let i = 0; i < r.length; i++) {
		let a = r[i];
		if (gi(t, e, a) && !ui(n, a)) return !0;
	}
	return !1;
}
function gi(e, t, n) {
	let r = e[n], i = t[n];
	return n === "style" && v(r) && v(i) ? !be(r, i) : r !== i;
}
function _i({ vnode: e, parent: t, suspense: n }, r) {
	for (; t;) {
		let n = t.subTree;
		if (n.suspense && n.suspense.activeBranch === e && (n.suspense.vnode.el = n.el = r, e = n), n === e) (e = t.vnode).el = r, t = t.parent;
		else break;
	}
	n && n.activeBranch === e && (n.vnode.el = r);
}
var vi = {}, yi = () => Object.create(vi), bi = (e) => Object.getPrototypeOf(e) === vi;
function xi(e, t, n, r = !1) {
	let i = {}, a = yi();
	e.propsDefaults = /* @__PURE__ */ Object.create(null), Ci(e, t, i, a);
	for (let t in e.propsOptions[0]) t in i || (i[t] = void 0);
	e.props = n ? r ? i : /* @__PURE__ */ Ft(i) : e.type.props ? i : a, e.attrs = a;
}
function Si(e, t, n, r) {
	let { props: i, attrs: a, vnode: { patchFlag: o } } = e, s = /* @__PURE__ */ P(i), [c] = e.propsOptions, l = !1;
	if ((r || o > 0) && !(o & 16)) {
		if (o & 8) {
			let n = e.vnode.dynamicProps;
			for (let r = 0; r < n.length; r++) {
				let o = n[r];
				if (ui(e.emitsOptions, o)) continue;
				let d = t[o];
				if (c) if (u(a, o)) d !== a[o] && (a[o] = d, l = !0);
				else {
					let t = T(o);
					i[t] = wi(c, s, t, d, e, !1);
				}
				else d !== a[o] && (a[o] = d, l = !0);
			}
		}
	} else {
		Ci(e, t, i, a) && (l = !0);
		let r;
		for (let a in s) (!t || !u(t, a) && ((r = E(a)) === a || !u(t, r))) && (c ? n && (n[a] !== void 0 || n[r] !== void 0) && (i[a] = wi(c, s, a, void 0, e, !0)) : delete i[a]);
		if (a !== s) for (let e in a) (!t || !u(t, e)) && (delete a[e], l = !0);
	}
	l && et(e.attrs, "set", "");
}
function Ci(e, n, r, i) {
	let [a, o] = e.propsOptions, s = !1, c;
	if (n) for (let t in n) {
		if (ee(t)) continue;
		let l = n[t], d;
		a && u(a, d = T(t)) ? !o || !o.includes(d) ? r[d] = l : (c ||= {})[d] = l : ui(e.emitsOptions, t) || (!(t in i) || l !== i[t]) && (i[t] = l, s = !0);
	}
	if (o) {
		let n = /* @__PURE__ */ P(r), i = c || t;
		for (let t = 0; t < o.length; t++) {
			let s = o[t];
			r[s] = wi(a, n, s, i[s], e, !u(i, s));
		}
	}
	return s;
}
function wi(e, t, n, r, i, a) {
	let o = e[n];
	if (o != null) {
		let e = u(o, "default");
		if (e && r === void 0) {
			let e = o.default;
			if (o.type !== Function && !o.skipFactory && h(e)) {
				let { propsDefaults: a } = i;
				if (n in a) r = a[n];
				else {
					let o = wa(i);
					r = a[n] = e.call(null, t), o();
				}
			} else r = e;
			i.ce && i.ce._setProp(n, r);
		}
		o[0] && (a && !e ? r = !1 : o[1] && (r === "" || r === E(n)) && (r = !0));
	}
	return r;
}
var Ti = /* @__PURE__ */ new WeakMap();
function Ei(e, r, i = !1) {
	let a = i ? Ti : r.propsCache, o = a.get(e);
	if (o) return o;
	let c = e.props, l = {}, f = [], p = !1;
	if (!h(e)) {
		let t = (e) => {
			p = !0;
			let [t, n] = Ei(e, r, !0);
			s(l, t), n && f.push(...n);
		};
		!i && r.mixins.length && r.mixins.forEach(t), e.extends && t(e.extends), e.mixins && e.mixins.forEach(t);
	}
	if (!c && !p) return v(e) && a.set(e, n), n;
	if (d(c)) for (let e = 0; e < c.length; e++) {
		let n = T(c[e]);
		Di(n) && (l[n] = t);
	}
	else if (c) for (let e in c) {
		let t = T(e);
		if (Di(t)) {
			let n = c[e], r = l[t] = d(n) || h(n) ? { type: n } : s({}, n), i = r.type, a = !1, o = !0;
			if (d(i)) for (let e = 0; e < i.length; ++e) {
				let t = i[e], n = h(t) && t.name;
				if (n === "Boolean") {
					a = !0;
					break;
				}
				n === "String" && (o = !1);
			}
			else a = h(i) && i.name === "Boolean";
			r[0] = a, r[1] = o, (a || u(r, "default")) && f.push(t);
		}
	}
	let m = [l, f];
	return v(e) && a.set(e, m), m;
}
function Di(e) {
	return e[0] !== "$" && !ee(e);
}
var Oi = (e) => e === "_" || e === "_ctx" || e === "$stable", ki = (e) => d(e) ? e.map(fa) : [fa(e)], Ai = (e, t, n) => {
	if (t._n) return t;
	let r = Dn((...e) => ki(t(...e)), n);
	return r._c = !1, r;
}, ji = (e, t, n) => {
	let r = e._ctx;
	for (let n in e) {
		if (Oi(n)) continue;
		let i = e[n];
		if (h(i)) t[n] = Ai(n, i, r);
		else if (i != null) {
			let e = ki(i);
			t[n] = () => e;
		}
	}
}, Mi = (e, t) => {
	let n = ki(t);
	e.slots.default = () => n;
}, Ni = (e, t, n) => {
	for (let r in t) (n || !Oi(r)) && (e[r] = t[r]);
}, Pi = (e, t, n) => {
	let r = e.slots = yi();
	if (e.vnode.shapeFlag & 32) {
		let e = t._;
		e ? (Ni(r, t, n), n && O(r, "_", e, !0)) : ji(t, r);
	} else t && Mi(e, t);
}, Fi = (e, n, r) => {
	let { vnode: i, slots: a } = e, o = !0, s = t;
	if (i.shapeFlag & 32) {
		let e = n._;
		e ? r && e === 1 ? o = !1 : Ni(a, n, r) : (o = !n.$stable, ji(n, a)), s = n;
	} else n && (Mi(e, n), s = { default: 1 });
	if (o) for (let e in a) !Oi(e) && s[e] == null && delete a[e];
}, V = qi;
function Ii(e) {
	return Li(e);
}
function Li(e, i) {
	let a = ue();
	a.__VUE__ = !0;
	let { insert: o, remove: s, patchProp: c, createElement: l, createText: u, createComment: d, setText: f, setElementText: p, parentNode: m, nextSibling: h, setScopeId: g = r, insertStaticContent: _ } = e, v = (e, t, n, r = null, i = null, a = null, o = void 0, s = null, c = !!t.dynamicChildren) => {
		if (e === t) return;
		e && !ia(e, t) && (r = ye(e), he(e, i, a, !0), e = null), t.patchFlag === -2 && (c = !1, t.dynamicChildren = null);
		let { type: l, ref: u, shapeFlag: d } = t;
		switch (l) {
			case Ji:
				y(e, t, n, r);
				break;
			case U:
				b(e, t, n, r);
				break;
			case Yi:
				e ?? x(t, n, r, o);
				break;
			case H:
				ae(e, t, n, r, i, a, o, s, c);
				break;
			default: d & 1 ? w(e, t, n, r, i, a, o, s, c) : d & 6 ? D(e, t, n, r, i, a, o, s, c) : (d & 64 || d & 128) && l.process(e, t, n, r, i, a, o, s, c, Se);
		}
		u != null && i ? gr(u, e && e.ref, a, t || e, !t) : u == null && e && e.ref != null && gr(e.ref, null, a, e, !0);
	}, y = (e, t, n, r) => {
		if (e == null) o(t.el = u(t.children), n, r);
		else {
			let n = t.el = e.el;
			t.children !== e.children && f(n, t.children);
		}
	}, b = (e, t, n, r) => {
		e == null ? o(t.el = d(t.children || ""), n, r) : t.el = e.el;
	}, x = (e, t, n, r) => {
		[e.el, e.anchor] = _(e.children, t, n, r, e.el, e.anchor);
	}, S = ({ el: e, anchor: t }, n, r) => {
		let i;
		for (; e && e !== t;) i = h(e), o(e, n, r), e = i;
		o(t, n, r);
	}, C = ({ el: e, anchor: t }) => {
		let n;
		for (; e && e !== t;) n = h(e), s(e), e = n;
		s(t);
	}, w = (e, t, n, r, i, a, o, s, c) => {
		if (t.type === "svg" ? o = "svg" : t.type === "math" && (o = "mathml"), e == null) te(t, n, r, i, a, o, s, c);
		else {
			let n = e.el && e.el._isVueCE ? e.el : null;
			try {
				n && n._beginPatch(), re(e, t, i, a, o, s, c);
			} finally {
				n && n._endPatch();
			}
		}
	}, te = (e, t, n, r, i, a, s, u) => {
		let d, f, { props: m, shapeFlag: h, transition: g, dirs: _ } = e;
		if (d = e.el = l(e.type, a, m && m.is, m), h & 8 ? p(d, e.children) : h & 16 && T(e.children, d, null, r, i, Ri(e, a), s, u), _ && On(e, null, r, "created"), ne(d, e, e.scopeId, s, r), m) {
			for (let e in m) e !== "value" && !ee(e) && c(d, e, null, m[e], a, r);
			"value" in m && c(d, "value", null, m.value, a), (f = m.onVnodeBeforeMount) && ga(f, r, e);
		}
		_ && On(e, null, r, "beforeMount");
		let v = Bi(i, g);
		v && g.beforeEnter(d), o(d, t, n), ((f = m && m.onVnodeMounted) || v || _) && V(() => {
			try {
				f && ga(f, r, e), v && g.enter(d), _ && On(e, null, r, "mounted");
			} finally {}
		}, i);
	}, ne = (e, t, n, r, i) => {
		if (n && g(e, n), r) for (let t = 0; t < r.length; t++) g(e, r[t]);
		if (i) {
			let n = i.subTree;
			if (t === n || Ki(n.type) && (n.ssContent === t || n.ssFallback === t)) {
				let t = i.vnode;
				ne(e, t, t.scopeId, t.slotScopeIds, i.parent);
			}
		}
	}, T = (e, t, n, r, i, a, o, s, c = 0) => {
		for (let l = c; l < e.length; l++) {
			let c = e[l] = s ? pa(e[l]) : fa(e[l]);
			v(null, c, t, n, r, i, a, o, s);
		}
	}, re = (e, n, r, i, a, o, s) => {
		let l = n.el = e.el, { patchFlag: u, dynamicChildren: d, dirs: f } = n;
		u |= e.patchFlag & 16;
		let m = e.props || t, h = n.props || t, g;
		if (r && zi(r, !1), (g = h.onVnodeBeforeUpdate) && ga(g, r, n, e), f && On(n, e, r, "beforeUpdate"), r && zi(r, !0), d && (!e.dynamicChildren || e.dynamicChildren.length !== d.length) && (u = 0, s = !1, d = null), (m.innerHTML && h.innerHTML == null || m.textContent && h.textContent == null) && p(l, ""), d ? E(e.dynamicChildren, d, l, r, i, Ri(n, a), o) : s || de(e, n, l, null, r, i, Ri(n, a), o, !1), u > 0) {
			if (u & 16) ie(l, m, h, r, a);
			else if (u & 2 && m.class !== h.class && c(l, "class", null, h.class, a), u & 4 && c(l, "style", m.style, h.style, a), u & 8) {
				let e = n.dynamicProps;
				for (let t = 0; t < e.length; t++) {
					let n = e[t], i = m[n], o = h[n];
					(o !== i || n === "value") && c(l, n, i, o, a, r);
				}
			}
			u & 1 && e.children !== n.children && p(l, n.children);
		} else !s && d == null && ie(l, m, h, r, a);
		((g = h.onVnodeUpdated) || f) && V(() => {
			g && ga(g, r, n, e), f && On(n, e, r, "updated");
		}, i);
	}, E = (e, t, n, r, i, a, o) => {
		for (let s = 0; s < t.length; s++) {
			let c = e[s], l = t[s], u = c.el && (c.type === H || !ia(c, l) || c.shapeFlag & 198) ? m(c.el) : n;
			v(c, l, u, null, r, i, a, o, !0);
		}
	}, ie = (e, n, r, i, a) => {
		if (n !== r) {
			if (n !== t) for (let t in n) !ee(t) && !(t in r) && c(e, t, n[t], null, a, i);
			for (let t in r) {
				if (ee(t)) continue;
				let o = r[t], s = n[t];
				o !== s && t !== "value" && c(e, t, s, o, a, i);
			}
			"value" in r && c(e, "value", n.value, r.value, a);
		}
	}, ae = (e, t, n, r, i, a, s, c, l) => {
		let d = t.el = e ? e.el : u(""), f = t.anchor = e ? e.anchor : u(""), { patchFlag: p, dynamicChildren: m, slotScopeIds: h } = t;
		h && (c = c ? c.concat(h) : h), e == null ? (o(d, n, r), o(f, n, r), T(t.children || [], n, f, i, a, s, c, l)) : p > 0 && p & 64 && m && e.dynamicChildren && e.dynamicChildren.length === m.length ? (E(e.dynamicChildren, m, n, i, a, s, c), (t.key != null || i && t === i.subTree) && Vi(e, t, !0)) : de(e, t, n, f, i, a, s, c, l);
	}, D = (e, t, n, r, i, a, o, s, c) => {
		t.slotScopeIds = s, e == null ? t.shapeFlag & 512 ? i.ctx.activate(t, n, r, o, c) : O(t, n, r, i, a, o, c) : se(e, t, c);
	}, O = (e, t, n, r, i, a, o) => {
		let s = e.component = ya(e, r, i);
		if (yr(e) && (s.ctx.renderer = Se), Oa(s, !1, o), s.asyncDep) {
			if (i && i.registerDep(s, ce, o), !e.el) {
				let r = s.subTree = q(U);
				b(null, r, t, n), e.placeholder = r.el;
			}
		} else ce(s, e, t, n, i, a, o);
	}, se = (e, t, n) => {
		let r = t.component = e.component;
		if (mi(e, t, n)) if (r.asyncDep && !r.asyncResolved) {
			le(r, t, n);
			return;
		} else r.next = t, r.update();
		else t.el = e.el, r.vnode = t;
	}, ce = (e, t, n, r, i, a, o) => {
		let s = () => {
			if (e.isMounted) {
				let { next: t, bu: n, u: r, parent: s, vnode: c } = e;
				{
					let n = Ui(e);
					if (n) {
						t && (t.el = c.el, le(e, t, o)), n.asyncDep.then(() => {
							V(() => {
								e.isUnmounted || l();
							}, i);
						});
						return;
					}
				}
				let u = t, d;
				zi(e, !1), t ? (t.el = c.el, le(e, t, o)) : t = c, n && oe(n), (d = t.props && t.props.onVnodeBeforeUpdate) && ga(d, s, t, c), zi(e, !0);
				let f = di(e), p = e.subTree;
				e.subTree = f, v(p, f, m(p.el), ye(p), e, i, a), t.el = f.el, u === null && _i(e, f.el), r && V(r, i), (d = t.props && t.props.onVnodeUpdated) && V(() => ga(d, s, t, c), i);
			} else {
				let o, { el: s, props: c } = t, { bm: l, m: u, parent: d, root: f, type: p } = e, m = vr(t);
				if (zi(e, !1), l && oe(l), !m && (o = c && c.onVnodeBeforeMount) && ga(o, d, t), zi(e, !0), s && Ce) {
					let t = () => {
						e.subTree = di(e), Ce(s, e.subTree, e, i, null);
					};
					m && p.__asyncHydrate ? p.__asyncHydrate(s, e, t) : t();
				} else {
					f.ce && f.ce._hasShadowRoot() && f.ce._injectChildStyle(p, e.parent ? e.parent.type : void 0);
					let o = e.subTree = di(e);
					v(null, o, n, r, e, i, a), t.el = o.el;
				}
				if (u && V(u, i), !m && (o = c && c.onVnodeMounted)) {
					let e = t;
					V(() => ga(o, d, e), i);
				}
				(t.shapeFlag & 256 || d && vr(d.vnode) && d.vnode.shapeFlag & 256) && e.a && V(e.a, i), e.isMounted = !0, t = n = r = null;
			}
		};
		e.scope.on();
		let c = e.effect = new Oe(s);
		e.scope.off();
		let l = e.update = c.run.bind(c), u = e.job = c.runIfDirty.bind(c);
		u.i = e, u.id = e.uid, c.scheduler = () => _n(u), zi(e, !0), l();
	}, le = (e, t, n) => {
		t.component = e;
		let r = e.vnode.props;
		e.vnode = t, e.next = null, Si(e, t.props, r, n), Fi(e, t.children, n), Ue(), bn(e), We();
	}, de = (e, t, n, r, i, a, o, s, c = !1) => {
		let l = e && e.children, u = e ? e.shapeFlag : 0, d = t.children, { patchFlag: f, shapeFlag: m } = t;
		if (f > 0) {
			if (f & 128) {
				pe(l, d, n, r, i, a, o, s, c);
				return;
			}
			if (f & 256) {
				fe(l, d, n, r, i, a, o, s, c);
				return;
			}
		}
		m & 8 ? (u & 16 && ve(l, i, a), d !== l && p(n, d)) : u & 16 ? m & 16 ? pe(l, d, n, r, i, a, o, s, c) : ve(l, i, a, !0) : (u & 8 && p(n, ""), m & 16 && T(d, n, r, i, a, o, s, c));
	}, fe = (e, t, r, i, a, o, s, c, l) => {
		e ||= n, t ||= n;
		let u = e.length, d = t.length, f = Math.min(u, d), p;
		for (p = 0; p < f; p++) {
			let n = t[p] = l ? pa(t[p]) : fa(t[p]);
			v(e[p], n, r, null, a, o, s, c, l);
		}
		u > d ? ve(e, a, o, !0, !1, f) : T(t, r, i, a, o, s, c, l, f);
	}, pe = (e, t, r, i, a, o, s, c, l) => {
		let u = 0, d = t.length, f = e.length - 1, p = d - 1;
		for (; u <= f && u <= p;) {
			let n = e[u], i = t[u] = l ? pa(t[u]) : fa(t[u]);
			if (ia(n, i)) v(n, i, r, null, a, o, s, c, l);
			else break;
			u++;
		}
		for (; u <= f && u <= p;) {
			let n = e[f], i = t[p] = l ? pa(t[p]) : fa(t[p]);
			if (ia(n, i)) v(n, i, r, null, a, o, s, c, l);
			else break;
			f--, p--;
		}
		if (u > f) {
			if (u <= p) {
				let e = p + 1, n = e < d ? t[e].el : i;
				for (; u <= p;) v(null, t[u] = l ? pa(t[u]) : fa(t[u]), r, n, a, o, s, c, l), u++;
			}
		} else if (u > p) for (; u <= f;) he(e[u], a, o, !0), u++;
		else {
			let m = u, h = u, g = /* @__PURE__ */ new Map();
			for (u = h; u <= p; u++) {
				let e = t[u] = l ? pa(t[u]) : fa(t[u]);
				e.key != null && g.set(e.key, u);
			}
			let _, y = 0, b = p - h + 1, x = !1, S = 0, C = Array(b);
			for (u = 0; u < b; u++) C[u] = 0;
			for (u = m; u <= f; u++) {
				let n = e[u];
				if (y >= b) {
					he(n, a, o, !0);
					continue;
				}
				let i;
				if (n.key != null) i = g.get(n.key);
				else for (_ = h; _ <= p; _++) if (C[_ - h] === 0 && ia(n, t[_])) {
					i = _;
					break;
				}
				i === void 0 ? he(n, a, o, !0) : (C[i - h] = u + 1, i >= S ? S = i : x = !0, v(n, t[i], r, null, a, o, s, c, l), y++);
			}
			let w = x ? Hi(C) : n;
			for (_ = w.length - 1, u = b - 1; u >= 0; u--) {
				let e = h + u, n = t[e], f = t[e + 1], p = e + 1 < d ? f.el || Gi(f) : i;
				C[u] === 0 ? v(null, n, r, p, a, o, s, c, l) : x && (_ < 0 || u !== w[_] ? me(n, r, p, 2) : _--);
			}
		}
	}, me = (e, t, n, r, i = null) => {
		let { el: a, type: c, transition: l, children: u, shapeFlag: d } = e;
		if (d & 6) {
			me(e.component.subTree, t, n, r);
			return;
		}
		if (d & 128) {
			e.suspense.move(t, n, r);
			return;
		}
		if (d & 64) {
			c.move(e, t, n, Se);
			return;
		}
		if (c === H) {
			o(a, t, n);
			for (let e = 0; e < u.length; e++) me(u[e], t, n, r);
			o(e.anchor, t, n);
			return;
		}
		if (c === Yi) {
			S(e, t, n);
			return;
		}
		if (r !== 2 && d & 1 && l) if (r === 0) l.persisted && !a[Zn] ? o(a, t, n) : (l.beforeEnter(a), o(a, t, n), V(() => l.enter(a), i));
		else {
			let { leave: r, delayLeave: i, afterLeave: c } = l, u = () => {
				e.ctx.isUnmounted ? s(a) : o(a, t, n);
			}, d = () => {
				let e = a._isLeaving || !!a[Zn];
				a._isLeaving && a[Zn](!0), l.persisted && !e ? u() : r(a, () => {
					u(), c && c();
				});
			};
			i ? i(a, u, d) : d();
		}
		else o(a, t, n);
	}, he = (e, t, n, r = !1, i = !1) => {
		let { type: a, props: o, ref: s, children: c, dynamicChildren: l, shapeFlag: u, patchFlag: d, dirs: f, cacheIndex: p, memo: m } = e;
		if (d === -2 && (i = !1), s != null && (Ue(), gr(s, null, n, e, !0), We()), p != null && (t.renderCache[p] = void 0), u & 256) {
			t.ctx.deactivate(e);
			return;
		}
		let h = u & 1 && f, g = !vr(e), _;
		if (g && (_ = o && o.onVnodeBeforeUnmount) && ga(_, t, e), u & 6) _e(e.component, n, r);
		else {
			if (u & 128) {
				e.suspense.unmount(n, r);
				return;
			}
			h && On(e, null, t, "beforeUnmount"), u & 64 ? e.type.remove(e, t, n, Se, r) : l && !l.hasOnce && (a !== H || d > 0 && d & 64) ? ve(l, t, n, !1, !0) : (a === H && d & 384 || !i && u & 16) && ve(c, t, n), r && k(e);
		}
		let v = m != null && p == null;
		(g && (_ = o && o.onVnodeUnmounted) || h || v) && V(() => {
			_ && ga(_, t, e), h && On(e, null, t, "unmounted"), v && (e.el = null);
		}, n);
	}, k = (e) => {
		let { type: t, el: n, anchor: r, transition: i } = e;
		if (t === H) {
			ge(n, r);
			return;
		}
		if (t === Yi) {
			C(e);
			return;
		}
		let a = () => {
			s(n), i && !i.persisted && i.afterLeave && i.afterLeave();
		};
		if (e.shapeFlag & 1 && i && !i.persisted) {
			let { leave: t, delayLeave: r } = i, o = () => t(n, a);
			r ? r(e.el, a, o) : o();
		} else a();
	}, ge = (e, t) => {
		let n;
		for (; e !== t;) n = h(e), s(e), e = n;
		s(t);
	}, _e = (e, t, n) => {
		let { bum: r, scope: i, job: a, subTree: o, um: s, m: c, a: l } = e;
		Wi(c), Wi(l), r && oe(r), i.stop(), a && (a.flags |= 8, he(o, e, t, n)), s && V(s, t), V(() => {
			e.isUnmounted = !0;
		}, t);
	}, ve = (e, t, n, r = !1, i = !1, a = 0) => {
		for (let o = a; o < e.length; o++) he(e[o], t, n, r, i);
	}, ye = (e) => {
		if (e.shapeFlag & 6) return ye(e.component.subTree);
		if (e.shapeFlag & 128) return e.suspense.next();
		let t = h(e.anchor || e.el), n = t && t[Rn];
		return n ? h(n) : t;
	}, be = !1, xe = (e, t, n) => {
		let r;
		e == null ? t._vnode && (he(t._vnode, null, null, !0), r = t._vnode.component) : v(t._vnode || null, e, t, null, null, null, n), t._vnode = e, be ||= (be = !0, bn(r), xn(), !1);
	}, Se = {
		p: v,
		um: he,
		m: me,
		r: k,
		mt: O,
		mc: T,
		pc: de,
		pbc: E,
		n: ye,
		o: e
	}, A, Ce;
	return i && ([A, Ce] = i(Se)), {
		render: xe,
		hydrate: A,
		createApp: ii(xe, A)
	};
}
function Ri({ type: e, props: t }, n) {
	return n === "svg" && e === "foreignObject" || n === "mathml" && e === "annotation-xml" && t && t.encoding && t.encoding.includes("html") ? void 0 : n;
}
function zi({ effect: e, job: t }, n) {
	n ? (e.flags |= 32, t.flags |= 4) : (e.flags &= -33, t.flags &= -5);
}
function Bi(e, t) {
	return (!e || e && !e.pendingBranch) && t && !t.persisted;
}
function Vi(e, t, n = !1) {
	let r = e.children, i = t.children;
	if (d(r) && d(i)) for (let e = 0; e < r.length; e++) {
		let t = r[e], a = i[e];
		a.shapeFlag & 1 && !a.dynamicChildren && ((a.patchFlag <= 0 || a.patchFlag === 32) && (a = i[e] = pa(i[e]), a.el = t.el), !n && a.patchFlag !== -2 && Vi(t, a)), a.type === Ji && (a.patchFlag === -1 && (a = i[e] = pa(a)), a.el = t.el), a.type === U && !a.el && (a.el = t.el);
	}
}
function Hi(e) {
	let t = e.slice(), n = [0], r, i, a, o, s, c = e.length;
	for (r = 0; r < c; r++) {
		let c = e[r];
		if (c !== 0) {
			if (i = n[n.length - 1], e[i] < c) {
				t[r] = i, n.push(r);
				continue;
			}
			for (a = 0, o = n.length - 1; a < o;) s = a + o >> 1, e[n[s]] < c ? a = s + 1 : o = s;
			c < e[n[a]] && (a > 0 && (t[r] = n[a - 1]), n[a] = r);
		}
	}
	for (a = n.length, o = n[a - 1]; a-- > 0;) n[a] = o, o = t[o];
	return n;
}
function Ui(e) {
	let t = e.subTree.component;
	if (t) return t.asyncDep && !t.asyncResolved ? t : Ui(t);
}
function Wi(e) {
	if (e) for (let t = 0; t < e.length; t++) e[t].flags |= 8;
}
function Gi(e) {
	if (e.placeholder) return e.placeholder;
	let t = e.component;
	return t ? Gi(t.subTree) : null;
}
var Ki = (e) => e.__isSuspense;
function qi(e, t) {
	t && t.pendingBranch ? d(e) ? t.effects.push(...e) : t.effects.push(e) : yn(e);
}
var H = /* @__PURE__ */ Symbol.for("v-fgt"), Ji = /* @__PURE__ */ Symbol.for("v-txt"), U = /* @__PURE__ */ Symbol.for("v-cmt"), Yi = /* @__PURE__ */ Symbol.for("v-stc"), Xi = [], Zi = null;
function W(e = !1) {
	Xi.push(Zi = e ? null : []);
}
function Qi() {
	Xi.pop(), Zi = Xi[Xi.length - 1] || null;
}
var $i = 1;
function ea(e, t = !1) {
	$i += e, e < 0 && Zi && t && (Zi.hasOnce = !0);
}
function ta(e) {
	return e.dynamicChildren = $i > 0 ? Zi || n : null, Qi(), $i > 0 && Zi && Zi.push(e), e;
}
function G(e, t, n, r, i, a) {
	return ta(K(e, t, n, r, i, a, !0));
}
function na(e, t, n, r, i) {
	return ta(q(e, t, n, r, i, !0));
}
function ra(e) {
	return e ? e.__v_isVNode === !0 : !1;
}
function ia(e, t) {
	return e.type === t.type && e.key === t.key;
}
var aa = ({ key: e }) => e ?? null, oa = ({ ref: e, ref_key: t, ref_for: n }) => (typeof e == "number" && (e = "" + e), e == null ? null : g(e) || /* @__PURE__ */ F(e) || h(e) ? {
	i: wn,
	r: e,
	k: t,
	f: !!n
} : e);
function K(e, t = null, n = null, r = 0, i = null, a = e === H ? 0 : 1, o = !1, s = !1) {
	let c = {
		__v_isVNode: !0,
		__v_skip: !0,
		type: e,
		props: t,
		key: t && aa(t),
		ref: t && oa(t),
		scopeId: Tn,
		slotScopeIds: null,
		children: n,
		component: null,
		suspense: null,
		ssContent: null,
		ssFallback: null,
		dirs: null,
		transition: null,
		el: null,
		anchor: null,
		target: null,
		targetStart: null,
		targetAnchor: null,
		staticCount: 0,
		shapeFlag: a,
		patchFlag: r,
		dynamicProps: i,
		dynamicChildren: null,
		appContext: null,
		ctx: wn
	};
	return s ? (ma(c, n), a & 128 && e.normalize(c)) : n && (c.shapeFlag |= g(n) ? 8 : 16), $i > 0 && !o && Zi && (c.patchFlag > 0 || a & 6) && c.patchFlag !== 32 && Zi.push(c), c;
}
var q = sa;
function sa(e, t = null, n = null, r = 0, i = null, a = !1) {
	if ((!e || e === Ir) && (e = U), ra(e)) {
		let r = la(e, t, !0);
		return n && ma(r, n), $i > 0 && !a && Zi && (r.shapeFlag & 6 ? Zi[Zi.indexOf(e)] = r : Zi.push(r)), r.patchFlag = -2, r;
	}
	if (La(e) && (e = e.__vccOpts), t) {
		t = ca(t);
		let { class: e, style: n } = t;
		e && !g(e) && (t.class = k(e)), v(n) && (/* @__PURE__ */ Vt(n) && !d(n) && (n = s({}, n)), t.style = de(n));
	}
	let o = g(e) ? 1 : Ki(e) ? 128 : zn(e) ? 64 : v(e) ? 4 : h(e) ? 2 : 0;
	return K(e, t, n, r, i, o, a, !0);
}
function ca(e) {
	return e ? /* @__PURE__ */ Vt(e) || bi(e) ? s({}, e) : e : null;
}
function la(e, t, n = !1, r = !1) {
	let { props: i, ref: a, patchFlag: o, children: s, transition: c } = e, l = t ? ha(i || {}, t) : i, u = {
		__v_isVNode: !0,
		__v_skip: !0,
		type: e.type,
		props: l,
		key: l && aa(l),
		ref: t && t.ref ? n && a ? d(a) ? a.concat(oa(t)) : [a, oa(t)] : oa(t) : a,
		scopeId: e.scopeId,
		slotScopeIds: e.slotScopeIds,
		children: s,
		target: e.target,
		targetStart: e.targetStart,
		targetAnchor: e.targetAnchor,
		staticCount: e.staticCount,
		shapeFlag: e.shapeFlag,
		patchFlag: t && e.type !== H ? o === -1 ? 16 : o | 16 : o,
		dynamicProps: e.dynamicProps,
		dynamicChildren: e.dynamicChildren,
		appContext: e.appContext,
		dirs: e.dirs,
		transition: c,
		component: e.component,
		suspense: e.suspense,
		ssContent: e.ssContent && la(e.ssContent),
		ssFallback: e.ssFallback && la(e.ssFallback),
		placeholder: e.placeholder,
		el: e.el,
		anchor: e.anchor,
		ctx: e.ctx,
		ce: e.ce
	};
	return c && r && ur(u, c.clone(u)), u;
}
function ua(e = " ", t = 0) {
	return q(Ji, null, e, t);
}
function da(e, t) {
	let n = q(Yi, null, e);
	return n.staticCount = t, n;
}
function J(e = "", t = !1) {
	return t ? (W(), na(U, null, e)) : q(U, null, e);
}
function fa(e) {
	return e == null || typeof e == "boolean" ? q(U) : d(e) ? q(H, null, e.slice()) : ra(e) ? pa(e) : q(Ji, null, String(e));
}
function pa(e) {
	return e.el === null && e.patchFlag !== -1 || e.memo ? e : la(e);
}
function ma(e, t) {
	let n = 0, { shapeFlag: r } = e;
	if (t == null) t = null;
	else if (d(t)) n = 16;
	else if (typeof t == "object") if (r & 65) {
		let n = t.default;
		n && (n._c && (n._d = !1), ma(e, n()), n._c && (n._d = !0));
		return;
	} else {
		n = 32;
		let r = t._;
		!r && !bi(t) ? t._ctx = wn : r === 3 && wn && (wn.slots._ === 1 ? t._ = 1 : (t._ = 2, e.patchFlag |= 1024));
	}
	else if (h(t)) {
		if (r & 65) {
			ma(e, { default: t });
			return;
		}
		t = {
			default: t,
			_ctx: wn
		}, n = 32;
	} else t = String(t), r & 64 ? (n = 16, t = [ua(t)]) : n = 8;
	e.children = t, e.shapeFlag |= n;
}
function ha(...e) {
	let t = {};
	for (let n = 0; n < e.length; n++) {
		let r = e[n];
		for (let e in r) if (e === "class") t.class !== r.class && (t.class = k([t.class, r.class]));
		else if (e === "style") t.style = de([t.style, r.style]);
		else if (a(e)) {
			let n = t[e], i = r[e];
			i && n !== i && !(d(n) && n.includes(i)) ? t[e] = n ? [].concat(n, i) : i : i == null && n == null && !o(e) && (t[e] = i);
		} else e !== "" && (t[e] = r[e]);
	}
	return t;
}
function ga(e, t, n, r = null) {
	on(e, t, 7, [n, r]);
}
var _a = ni(), va = 0;
function ya(e, n, r) {
	let i = e.type, a = (n ? n.appContext : e.appContext) || _a, o = {
		uid: va++,
		vnode: e,
		type: i,
		parent: n,
		appContext: a,
		root: null,
		next: null,
		subTree: null,
		effect: null,
		update: null,
		job: null,
		scope: new Te(!0),
		render: null,
		proxy: null,
		exposed: null,
		exposeProxy: null,
		withProxy: null,
		provides: n ? n.provides : Object.create(a.provides),
		ids: n ? n.ids : [
			"",
			0,
			0
		],
		accessCache: null,
		renderCache: [],
		components: null,
		directives: null,
		propsOptions: Ei(i, a),
		emitsOptions: li(i, a),
		emit: null,
		emitted: null,
		propsDefaults: t,
		inheritAttrs: i.inheritAttrs,
		ctx: t,
		data: t,
		props: t,
		attrs: t,
		slots: t,
		refs: t,
		setupState: t,
		setupContext: null,
		suspense: r,
		suspenseId: r ? r.pendingId : 0,
		asyncDep: null,
		asyncResolved: !1,
		isMounted: !1,
		isUnmounted: !1,
		isDeactivated: !1,
		bc: null,
		c: null,
		bm: null,
		m: null,
		bu: null,
		u: null,
		um: null,
		bum: null,
		da: null,
		a: null,
		rtg: null,
		rtc: null,
		ec: null,
		sp: null
	};
	return o.ctx = { _: o }, o.root = n ? n.root : o, o.emit = si.bind(null, o), e.ce && e.ce(o), o;
}
var ba = null, xa = () => ba || wn, Sa, Ca;
{
	let e = ue(), t = (t, n) => {
		let r;
		return (r = e[t]) || (r = e[t] = []), r.push(n), (e) => {
			r.length > 1 ? r.forEach((t) => t(e)) : r[0](e);
		};
	};
	Sa = t("__VUE_INSTANCE_SETTERS__", (e) => ba = e), Ca = t("__VUE_SSR_SETTERS__", (e) => Da = e);
}
var wa = (e) => {
	let t = ba;
	return Sa(e), e.scope.on(), () => {
		e.scope.off(), Sa(t);
	};
}, Ta = () => {
	ba && ba.scope.off(), Sa(null);
};
function Ea(e) {
	return e.vnode.shapeFlag & 4;
}
var Da = !1;
function Oa(e, t = !1, n = !1) {
	t && Ca(t);
	let { props: r, children: i } = e.vnode, a = Ea(e);
	xi(e, r, a, t), Pi(e, i, n || t);
	let o = a ? ka(e, t) : void 0;
	return t && Ca(!1), o;
}
function ka(e, t) {
	let n = e.type;
	e.accessCache = /* @__PURE__ */ Object.create(null), e.proxy = new Proxy(e.ctx, Br);
	let { setup: r } = n;
	if (r) {
		Ue();
		let n = e.setupContext = r.length > 1 ? Fa(e) : null, i = wa(e), a = an(r, e, 0, [e.props, n]), o = y(a);
		if (We(), i(), (o || e.sp) && !vr(e) && pr(e), o) {
			if (a.then(Ta, Ta), t) return a.then((n) => {
				Aa(e, n, t);
			}).catch((t) => {
				sn(t, e, 0);
			});
			e.asyncDep = a;
		} else Aa(e, a, t);
	} else Na(e, t);
}
function Aa(e, t, n) {
	h(t) ? e.type.__ssrInlineRender ? e.ssrRender = t : e.render = t : v(t) && (e.setupState = Yt(t)), Na(e, n);
}
var ja, Ma;
function Na(e, t, n) {
	let i = e.type;
	if (!e.render) {
		if (!t && ja && !i.render) {
			let t = i.template || qr(e).template;
			if (t) {
				let { isCustomElement: n, compilerOptions: r } = e.appContext.config, { delimiters: a, compilerOptions: o } = i;
				i.render = ja(t, s(s({
					isCustomElement: n,
					delimiters: a
				}, r), o));
			}
		}
		e.render = i.render || r, Ma && Ma(e);
	}
	{
		let t = wa(e);
		Ue();
		try {
			Ur(e);
		} finally {
			We(), t();
		}
	}
}
var Pa = { get(e, t) {
	return N(e, "get", ""), e[t];
} };
function Fa(e) {
	return {
		attrs: new Proxy(e.attrs, Pa),
		slots: e.slots,
		emit: e.emit,
		expose: (t) => {
			e.exposed = t || {};
		}
	};
}
function Ia(e) {
	return e.exposed ? e.exposeProxy ||= new Proxy(Yt(Ht(e.exposed)), {
		get(t, n) {
			if (n in t) return t[n];
			if (n in Rr) return Rr[n](e);
		},
		has(e, t) {
			return t in e || t in Rr;
		}
	}) : e.proxy;
}
function La(e) {
	return h(e) && "__vccOpts" in e;
}
var Y = (e, t) => /* @__PURE__ */ Zt(e, t, Da);
function Ra(e, t, n) {
	try {
		ea(-1);
		let r = arguments.length;
		return r === 2 ? v(t) && !d(t) ? ra(t) ? q(e, null, [t]) : q(e, t) : q(e, null, t) : (r > 3 ? n = Array.prototype.slice.call(arguments, 2) : r === 3 && ra(n) && (n = [n]), q(e, t, n));
	} finally {
		ea(1);
	}
}
var za = "3.5.40", Ba = void 0, Va = typeof window < "u" && window.trustedTypes;
if (Va) try {
	Ba = /* @__PURE__ */ Va.createPolicy("vue", { createHTML: (e) => e });
} catch {}
var Ha = Ba ? (e) => Ba.createHTML(e) : (e) => e, Ua = "http://www.w3.org/2000/svg", Wa = "http://www.w3.org/1998/Math/MathML", Ga = typeof document < "u" ? document : null, Ka = Ga && /* @__PURE__ */ Ga.createElement("template"), qa = {
	insert: (e, t, n) => {
		t.insertBefore(e, n || null);
	},
	remove: (e) => {
		let t = e.parentNode;
		t && t.removeChild(e);
	},
	createElement: (e, t, n, r) => {
		let i = t === "svg" ? Ga.createElementNS(Ua, e) : t === "mathml" ? Ga.createElementNS(Wa, e) : n ? Ga.createElement(e, { is: n }) : Ga.createElement(e);
		return e === "select" && r && r.multiple != null && i.setAttribute("multiple", r.multiple), i;
	},
	createText: (e) => Ga.createTextNode(e),
	createComment: (e) => Ga.createComment(e),
	setText: (e, t) => {
		e.nodeValue = t;
	},
	setElementText: (e, t) => {
		e.textContent = t;
	},
	parentNode: (e) => e.parentNode,
	nextSibling: (e) => e.nextSibling,
	querySelector: (e) => Ga.querySelector(e),
	setScopeId(e, t) {
		e.setAttribute(t, "");
	},
	insertStaticContent(e, t, n, r, i, a) {
		let o = n ? n.previousSibling : t.lastChild;
		if (i && (i === a || i.nextSibling)) for (; t.insertBefore(i.cloneNode(!0), n), !(i === a || !(i = i.nextSibling)););
		else {
			Ka.innerHTML = Ha(r === "svg" ? `<svg>${e}</svg>` : r === "mathml" ? `<math>${e}</math>` : e);
			let i = Ka.content;
			if (r === "svg" || r === "mathml") {
				let e = i.firstChild;
				for (; e.firstChild;) i.appendChild(e.firstChild);
				i.removeChild(e);
			}
			t.insertBefore(i, n);
		}
		return [o ? o.nextSibling : t.firstChild, n ? n.previousSibling : t.lastChild];
	}
}, Ja = "transition", Ya = "animation", Xa = /* @__PURE__ */ Symbol("_vtc"), Za = {
	name: String,
	type: String,
	css: {
		type: Boolean,
		default: !0
	},
	duration: [
		String,
		Number,
		Object
	],
	enterFromClass: String,
	enterActiveClass: String,
	enterToClass: String,
	appearFromClass: String,
	appearActiveClass: String,
	appearToClass: String,
	leaveFromClass: String,
	leaveActiveClass: String,
	leaveToClass: String
}, Qa = /* @__PURE__ */ s({}, tr, Za), $a = /* @__PURE__ */ ((e) => (e.displayName = "Transition", e.props = Qa, e))((e, { slots: t }) => Ra(ar, no(e), t)), eo = (e, t = []) => {
	d(e) ? e.forEach((e) => e(...t)) : e && e(...t);
}, to = (e) => e ? d(e) ? e.some((e) => e.length > 1) : e.length > 1 : !1;
function no(e) {
	let t = {};
	for (let n in e) n in Za || (t[n] = e[n]);
	if (e.css === !1) return t;
	let { name: n = "v", type: r, duration: i, enterFromClass: a = `${n}-enter-from`, enterActiveClass: o = `${n}-enter-active`, enterToClass: c = `${n}-enter-to`, appearFromClass: l = a, appearActiveClass: u = o, appearToClass: d = c, leaveFromClass: f = `${n}-leave-from`, leaveActiveClass: p = `${n}-leave-active`, leaveToClass: m = `${n}-leave-to` } = e, h = ro(i), g = h && h[0], _ = h && h[1], { onBeforeEnter: v, onEnter: y, onEnterCancelled: b, onLeave: x, onLeaveCancelled: S, onBeforeAppear: C = v, onAppear: w = y, onAppearCancelled: ee = b } = t, te = (e, t, n, r) => {
		e._enterCancelled = r, oo(e, t ? d : c), oo(e, t ? u : o), n && n();
	}, ne = (e, t) => {
		e._isLeaving = !1, oo(e, f), oo(e, m), oo(e, p), t && t();
	}, T = (e) => (t, n) => {
		let i = e ? w : y, o = () => te(t, e, n);
		eo(i, [t, o]), so(() => {
			oo(t, e ? l : a), ao(t, e ? d : c), to(i) || lo(t, r, g, o);
		});
	};
	return s(t, {
		onBeforeEnter(e) {
			eo(v, [e]), ao(e, a), ao(e, o);
		},
		onBeforeAppear(e) {
			eo(C, [e]), ao(e, l), ao(e, u);
		},
		onEnter: T(!1),
		onAppear: T(!0),
		onLeave(e, t) {
			e._isLeaving = !0;
			let n = () => ne(e, t);
			ao(e, f), e._enterCancelled ? (ao(e, p), mo(e)) : (mo(e), ao(e, p)), so(() => {
				e._isLeaving && (oo(e, f), ao(e, m), to(x) || lo(e, r, _, n));
			}), eo(x, [e, n]);
		},
		onEnterCancelled(e) {
			te(e, !1, void 0, !0), eo(b, [e]);
		},
		onAppearCancelled(e) {
			te(e, !0, void 0, !0), eo(ee, [e]);
		},
		onLeaveCancelled(e) {
			ne(e), eo(S, [e]);
		}
	});
}
function ro(e) {
	if (e == null) return null;
	if (v(e)) return [io(e.enter), io(e.leave)];
	{
		let t = io(e);
		return [t, t];
	}
}
function io(e) {
	return ce(e);
}
function ao(e, t) {
	t.split(/\s+/).forEach((t) => t && e.classList.add(t)), (e[Xa] || (e[Xa] = /* @__PURE__ */ new Set())).add(t);
}
function oo(e, t) {
	t.split(/\s+/).forEach((t) => t && e.classList.remove(t));
	let n = e[Xa];
	n && (n.delete(t), n.size || (e[Xa] = void 0));
}
function so(e) {
	requestAnimationFrame(() => {
		requestAnimationFrame(e);
	});
}
var co = 0;
function lo(e, t, n, r) {
	let i = e._endId = ++co, a = () => {
		i === e._endId && r();
	};
	if (n != null) return setTimeout(a, n);
	let { type: o, timeout: s, propCount: c } = uo(e, t);
	if (!o) return r();
	let l = o + "end", u = 0, d = () => {
		e.removeEventListener(l, f), a();
	}, f = (t) => {
		t.target === e && ++u >= c && d();
	};
	setTimeout(() => {
		u < c && d();
	}, s + 1), e.addEventListener(l, f);
}
function uo(e, t) {
	let n = window.getComputedStyle(e), r = (e) => (n[e] || "").split(", "), i = r(`${Ja}Delay`), a = r(`${Ja}Duration`), o = fo(i, a), s = r(`${Ya}Delay`), c = r(`${Ya}Duration`), l = fo(s, c), u = null, d = 0, f = 0;
	t === Ja ? o > 0 && (u = Ja, d = o, f = a.length) : t === Ya ? l > 0 && (u = Ya, d = l, f = c.length) : (d = Math.max(o, l), u = d > 0 ? o > l ? Ja : Ya : null, f = u ? u === Ja ? a.length : c.length : 0);
	let p = u === Ja && /\b(?:transform|all)(?:,|$)/.test(r(`${Ja}Property`).toString());
	return {
		type: u,
		timeout: d,
		propCount: f,
		hasTransform: p
	};
}
function fo(e, t) {
	for (; e.length < t.length;) e = e.concat(e);
	return Math.max(...t.map((t, n) => po(t) + po(e[n])));
}
function po(e) {
	return e === "auto" ? 0 : Number(e.slice(0, -1).replace(",", ".")) * 1e3;
}
function mo(e) {
	return (e ? e.ownerDocument : document).body.offsetHeight;
}
function ho(e, t, n) {
	let r = e[Xa];
	r && (t = (t ? [t, ...r] : [...r]).join(" ")), t == null ? e.removeAttribute("class") : n ? e.setAttribute("class", t) : e.className = t;
}
var go = /* @__PURE__ */ Symbol("_vod"), _o = /* @__PURE__ */ Symbol("_vsh"), vo = /* @__PURE__ */ Symbol(""), yo = /(?:^|;)\s*display\s*:/;
function bo(e, t, n) {
	let r = e.style, i = g(n), a = !1;
	if (n && !i) {
		if (t) if (g(t)) for (let e of t.split(";")) {
			let t = e.slice(0, e.indexOf(":")).trim();
			n[t] ?? So(r, t, "");
		}
		else for (let e in t) n[e] ?? So(r, e, "");
		for (let i in n) {
			i === "display" && (a = !0);
			let o = n[i];
			o == null ? So(r, i, "") : Eo(e, i, !g(t) && t ? t[i] : void 0, o) || So(r, i, o);
		}
	} else if (i) {
		if (t !== n) {
			let e = r[vo];
			e && (n += ";" + e), r.cssText = n, a = yo.test(n);
		}
	} else t && e.removeAttribute("style");
	go in e && (e[go] = a ? r.display : "", e[_o] && (r.display = "none"));
}
var xo = /\s*!important$/;
function So(e, t, n) {
	if (d(n)) n.forEach((n) => So(e, t, n));
	else if (n ??= "", t.startsWith("--")) e.setProperty(t, n);
	else {
		let r = To(e, t);
		xo.test(n) ? e.setProperty(E(r), n.replace(xo, ""), "important") : e[r] = n;
	}
}
var Co = [
	"Webkit",
	"Moz",
	"ms"
], wo = {};
function To(e, t) {
	let n = wo[t];
	if (n) return n;
	let r = T(t);
	if (r !== "filter" && r in e) return wo[t] = r;
	r = ie(r);
	for (let n = 0; n < Co.length; n++) {
		let i = Co[n] + r;
		if (i in e) return wo[t] = i;
	}
	return t;
}
function Eo(e, t, n, r) {
	return e.tagName === "TEXTAREA" && (t === "width" || t === "height") && g(r) && n === r;
}
var Do = "http://www.w3.org/1999/xlink";
function Oo(e, t, n, r, i, a = _e(t)) {
	r && t.startsWith("xlink:") ? n == null ? e.removeAttributeNS(Do, t.slice(6, t.length)) : e.setAttributeNS(Do, t, n) : n == null || a && !ve(n) ? e.removeAttribute(t) : e.setAttribute(t, a ? "" : _(n) ? String(n) : n);
}
function ko(e, t, n, r, i) {
	if (t === "innerHTML" || t === "textContent") {
		n != null && (e[t] = t === "innerHTML" ? Ha(n) : n);
		return;
	}
	let a = e.tagName;
	if (t === "value" && a !== "PROGRESS" && !a.includes("-")) {
		let r = a === "OPTION" ? e.getAttribute("value") || "" : e.value, i = n == null ? e.type === "checkbox" ? "on" : "" : String(n);
		(r !== i || !("_value" in e)) && (e.value = i), n ?? e.removeAttribute(t), e._value = n;
		return;
	}
	let o = !1;
	if (n === "" || n == null) {
		let r = typeof e[t];
		r === "boolean" ? n = ve(n) : n == null && r === "string" ? (n = "", o = !0) : r === "number" && (n = 0, o = !0);
	}
	try {
		e[t] = n;
	} catch {}
	o && e.removeAttribute(i || t);
}
function Ao(e, t, n, r) {
	e.addEventListener(t, n, r);
}
function jo(e, t, n, r) {
	e.removeEventListener(t, n, r);
}
var Mo = /* @__PURE__ */ Symbol("_vei");
function No(e, t, n, r, i = null) {
	let a = e[Mo] || (e[Mo] = {}), o = a[t];
	if (r && o) o.value = r;
	else {
		let [n, s] = Io(t);
		r ? Ao(e, n, a[t] = Bo(r, i), s) : o && (jo(e, n, o, s), a[t] = void 0);
	}
}
var Po = /(Once|Passive|Capture)$/, Fo = /^on:?(?:Once|Passive|Capture)$/;
function Io(e) {
	let t, n;
	for (; (n = e.match(Po)) && !Fo.test(e);) t ||= {}, e = e.slice(0, e.length - n[1].length), t[n[1].toLowerCase()] = !0;
	return [e[2] === ":" ? e.slice(3) : E(e.slice(2)), t];
}
var Lo = 0, Ro = /* @__PURE__ */ Promise.resolve(), zo = () => Lo ||= (Ro.then(() => Lo = 0), Date.now());
function Bo(e, t) {
	let n = (e) => {
		if (!e._vts) e._vts = Date.now();
		else if (e._vts <= n.attached) return;
		let r = n.value;
		if (d(r)) {
			let n = e.stopImmediatePropagation;
			e.stopImmediatePropagation = () => {
				n.call(e), e._stopped = !0;
			};
			let i = r.slice(), a = [e];
			for (let n = 0; n < i.length && !e._stopped; n++) {
				let e = i[n];
				e && on(e, t, 5, a);
			}
		} else on(r, t, 5, [e]);
	};
	return n.value = e, n.attached = zo(), n;
}
var Vo = (e) => e.charCodeAt(0) === 111 && e.charCodeAt(1) === 110 && e.charCodeAt(2) > 96 && e.charCodeAt(2) < 123, Ho = (e, t, n, r, i, s) => {
	let c = i === "svg";
	t === "class" ? ho(e, r, c) : t === "style" ? bo(e, n, r) : a(t) ? o(t) || No(e, t, n, r, s) : (t[0] === "." ? (t = t.slice(1), !0) : t[0] === "^" ? (t = t.slice(1), !1) : Uo(e, t, r, c)) ? (ko(e, t, r), !e.tagName.includes("-") && (t === "value" || t === "checked" || t === "selected") && Oo(e, t, r, c, s, t !== "value")) : e._isVueCE && (Wo(e, t) || e._def.__asyncLoader && (/[A-Z]/.test(t) || !g(r))) ? ko(e, T(t), r, s, t) : (t === "true-value" ? e._trueValue = r : t === "false-value" && (e._falseValue = r), Oo(e, t, r, c));
};
function Uo(e, t, n, r) {
	if (r) return !!(t === "innerHTML" || t === "textContent" || t in e && Vo(t) && h(n));
	if (t === "spellcheck" || t === "draggable" || t === "translate" || t === "autocorrect" || t === "sandbox" && e.tagName === "IFRAME" || t === "form" || t === "list" && e.tagName === "INPUT" || t === "type" && e.tagName === "TEXTAREA") return !1;
	if (t === "width" || t === "height") {
		let t = e.tagName;
		if (t === "IMG" || t === "VIDEO" || t === "CANVAS" || t === "SOURCE") return !1;
	}
	return Vo(t) && g(n) ? !1 : t in e;
}
function Wo(e, t) {
	let n = e._def.props;
	if (!n) return !1;
	let r = T(t);
	return Array.isArray(n) ? n.some((e) => T(e) === r) : Object.keys(n).some((e) => T(e) === r);
}
var Go = (e) => {
	let t = e.props["onUpdate:modelValue"] || !1;
	return d(t) ? (e) => oe(t, e) : t;
};
function Ko(e) {
	e.target.composing = !0;
}
function qo(e) {
	let t = e.target;
	t.composing && (t.composing = !1, t.dispatchEvent(new Event("input")));
}
var Jo = /* @__PURE__ */ Symbol("_assign");
function Yo(e, t, n) {
	return t && (e = e.trim()), n && (e = se(e)), e;
}
var Xo = {
	created(e, { modifiers: { lazy: t, trim: n, number: r } }, i) {
		e[Jo] = Go(i);
		let a = r || i.props && i.props.type === "number";
		Ao(e, t ? "change" : "input", (t) => {
			t.target.composing || e[Jo](Yo(e.value, n, a));
		}), (n || a) && Ao(e, "change", () => {
			e.value = Yo(e.value, n, a);
		}), t || (Ao(e, "compositionstart", Ko), Ao(e, "compositionend", qo), Ao(e, "change", qo));
	},
	mounted(e, { value: t }) {
		e.value = t ?? "";
	},
	beforeUpdate(e, { value: t, oldValue: n, modifiers: { lazy: r, trim: i, number: a } }, o) {
		if (e[Jo] = Go(o), e.composing) return;
		let s = (a || e.type === "number") && !/^0\d/.test(e.value) ? se(e.value) : e.value, c = t ?? "";
		if (s === c) return;
		let l = e.getRootNode();
		(l instanceof Document || l instanceof ShadowRoot) && l.activeElement === e && e.type !== "range" && (r && t === n || i && e.value.trim() === c) || (e.value = c);
	}
}, Zo = {
	deep: !0,
	created(e, t, n) {
		e[Jo] = Go(n), Ao(e, "change", () => {
			let t = e._modelValue, n = ts(e), r = e.checked, i = e[Jo];
			if (d(t)) {
				let e = xe(t, n), a = e !== -1;
				if (r && !a) i(t.concat(n));
				else if (!r && a) {
					let n = [...t];
					n.splice(e, 1), i(n);
				}
			} else if (p(t)) {
				let e = new Set(t);
				r ? e.add(n) : e.delete(n), i(e);
			} else i(ns(e, r));
		});
	},
	mounted: Qo,
	beforeUpdate(e, t, n) {
		e[Jo] = Go(n), Qo(e, t, n);
	}
};
function Qo(e, { value: t, oldValue: n }, r) {
	e._modelValue = t;
	let i;
	if (d(t)) i = xe(t, r.props.value) > -1;
	else if (p(t)) i = t.has(r.props.value);
	else {
		if (t === n) return;
		i = be(t, ns(e, !0));
	}
	e.checked !== i && (e.checked = i);
}
var $o = {
	deep: !0,
	created(e, { value: t, modifiers: { number: n } }, r) {
		e._modelValue = t, Ao(e, "change", () => {
			let t = Array.prototype.filter.call(e.options, (e) => e.selected).map((e) => n ? se(ts(e)) : ts(e));
			e[Jo](e.multiple ? p(e._modelValue) ? new Set(t) : t : t[0]), e._assigning = !0, hn(() => {
				e._assigning = !1;
			});
		}), e[Jo] = Go(r);
	},
	mounted(e, { value: t }) {
		es(e, t);
	},
	beforeUpdate(e, { value: t }, n) {
		e._modelValue = t, e[Jo] = Go(n);
	},
	updated(e, { value: t }) {
		e._assigning || es(e, t);
	}
};
function es(e, t) {
	let n = e.multiple, r = d(t);
	if (!(n && !r && !p(t))) {
		for (let i = 0, a = e.options.length; i < a; i++) {
			let a = e.options[i], o = ts(a);
			if (n) if (r) {
				let e = typeof o;
				a.selected = e === "string" || e === "number" ? t.some((e) => String(e) === String(o)) : xe(t, o) > -1;
			} else a.selected = t.has(o);
			else if (be(ts(a), t)) {
				e.selectedIndex !== i && (e.selectedIndex = i);
				return;
			}
		}
		!n && e.selectedIndex !== -1 && (e.selectedIndex = -1);
	}
}
function ts(e) {
	return "_value" in e ? e._value : e.value;
}
function ns(e, t) {
	let n = t ? "_trueValue" : "_falseValue";
	return n in e ? e[n] : t;
}
var rs = [
	"ctrl",
	"shift",
	"alt",
	"meta"
], is = {
	stop: (e) => e.stopPropagation(),
	prevent: (e) => e.preventDefault(),
	self: (e) => e.target !== e.currentTarget,
	ctrl: (e) => !e.ctrlKey,
	shift: (e) => !e.shiftKey,
	alt: (e) => !e.altKey,
	meta: (e) => !e.metaKey,
	left: (e) => "button" in e && e.button !== 0,
	middle: (e) => "button" in e && e.button !== 1,
	right: (e) => "button" in e && e.button !== 2,
	exact: (e, t) => rs.some((n) => e[`${n}Key`] && !t.includes(n))
}, as = (e, t) => {
	if (!e) return e;
	let n = e._withMods ||= {}, r = t.join(".");
	return n[r] || (n[r] = ((n, ...r) => {
		for (let e = 0; e < t.length; e++) {
			let r = is[t[e]];
			if (r && r(n, t)) return;
		}
		return e(n, ...r);
	}));
}, os = {
	esc: "escape",
	space: " ",
	up: "arrow-up",
	left: "arrow-left",
	right: "arrow-right",
	down: "arrow-down",
	delete: "backspace"
}, ss = (e, t) => {
	let n = e._withKeys ||= {}, r = t.join(".");
	return n[r] || (n[r] = ((n) => {
		if (!("key" in n)) return;
		let r = E(n.key);
		if (t.some((e) => e === r || os[e] === r)) return e(n);
	}));
}, cs = /* @__PURE__ */ s({ patchProp: Ho }, qa), ls;
function us() {
	return ls ||= Ii(cs);
}
var ds = ((...e) => {
	let t = us().createApp(...e), { mount: n } = t;
	return t.mount = (e) => {
		let r = ps(e);
		if (!r) return;
		let i = t._component;
		!h(i) && !i.render && !i.template && (i.template = r.innerHTML), r.nodeType === 1 && (r.textContent = "");
		let a = n(r, !1, fs(r));
		return r instanceof Element && (r.removeAttribute("v-cloak"), r.setAttribute("data-v-app", "")), a;
	}, t;
});
function fs(e) {
	if (e instanceof SVGElement) return "svg";
	if (typeof MathMLElement == "function" && e instanceof MathMLElement) return "mathml";
}
function ps(e) {
	return g(e) ? document.querySelector(e) : e;
}
//#endregion
//#region src/api.ts
async function ms(e, t = {}) {
	let n = await fetch(e, {
		credentials: "same-origin",
		...t,
		headers: {
			Accept: "application/json",
			...t.headers || {}
		}
	}), r = (n.headers.get("content-type") || "").includes("application/json") ? await n.json() : await n.text();
	if (!n.ok) {
		let e = typeof r == "object" && r ? r.error || r.detail || r.message : r;
		throw Error(String(e || `Request failed (${n.status})`));
	}
	return r;
}
function hs(e) {
	return ms(e);
}
function gs(e, t = {}) {
	return ms(e, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(t)
	});
}
function _s(e, t) {
	return ms(e, {
		method: "PUT",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(t)
	});
}
//#endregion
//#region src/trainerStore.ts
var vs = () => ({
	running: !1,
	exit_code: null,
	log_lines: []
}), ys = () => ({
	personal: [],
	negative: [],
	personal_count: 0,
	negative_count: 0
}), bs = () => ({
	items: [],
	captured_count: 0,
	personal_count: 0,
	negative_count: 0
}), xs = () => ({
	items: [],
	total_size_bytes: 0,
	total_file_count: 0
}), Ss = () => ({
	enabled: !1,
	wake_phrase: "",
	language: "en",
	english_accent: "mixed",
	stt_engine: "faster_whisper",
	minimum_transcript_chars: 2,
	delete_confirmed_wakes: !1,
	promote_close_misses: !1,
	schedule_hours: 24,
	minimum_new_negatives: 3,
	advertised_base_url: "",
	tater_url: "http://127.0.0.1:8501",
	notify_satellites: !0,
	train_microwakeword: !0,
	train_openwakeword: !1,
	train_nanowakeword_cnn: !1,
	train_nanowakeword_gru: !1,
	generated_negative_phrases: ""
}), X = /* @__PURE__ */ Pt({
	activeView: "trainer",
	initialized: !1,
	busy: /* @__PURE__ */ new Set(),
	phrase: "",
	language: "en",
	englishAccent: "mixed",
	ttsMode: "hybrid",
	trainMicroWakeWord: !0,
	trainOpenWakeWord: !1,
	trainNanoWakeWordCnn: !1,
	trainNanoWakeWordGru: !1,
	generatedNegativePhrases: "",
	languages: [{
		code: "en",
		label: "English (en)",
		engines: ["omnivoice"]
	}],
	englishAccents: [
		{
			code: "mixed",
			label: "Mixed English"
		},
		{
			code: "australian",
			label: "Australian"
		},
		{
			code: "american",
			label: "American"
		},
		{
			code: "british",
			label: "British"
		},
		{
			code: "canadian",
			label: "Canadian"
		},
		{
			code: "irish",
			label: "Irish"
		},
		{
			code: "scottish",
			label: "Scottish"
		},
		{
			code: "new_zealand",
			label: "New Zealand"
		},
		{
			code: "indian",
			label: "Indian"
		},
		{
			code: "south_african",
			label: "South African"
		}
	],
	session: {},
	samples: ys(),
	captured: bs(),
	training: vs(),
	auto: {},
	autoForm: Ss(),
	wakeWords: [],
	managedData: xs(),
	selectedFiles: [],
	sampleBucket: "personal",
	samplePage: {
		personal: 0,
		negative: 0
	},
	uploadProgress: 0,
	uploadLabel: "No upload in progress",
	uploadDetail: "Choose files and upload when you are ready.",
	consoleOpen: !1,
	taterLinkOpen: !1,
	trimItem: null,
	trimBucket: "personal",
	toast: {
		message: "",
		tone: "success",
		serial: 0
	}
}), Cs = 0, ws = 0, Ts = Y(() => Number(X.samples.personal_count ?? X.session.takes_received ?? 0)), Es = Y(() => Number(X.samples.negative_count ?? X.captured.negative_count ?? 0)), Ds = Y(() => X.languages.find((e) => e.code === X.language) || X.languages[0]), Os = Y(() => {
	let e = Ds.value?.engines?.length ? Ds.value.engines : ["omnivoice"], t = X.ttsMode === "piper" ? e.filter((e) => e === "piper") : X.ttsMode === "hybrid" ? e : e.filter((e) => e !== "piper"), n = {
		omnivoice: "OmniVoice",
		qwen3: "Qwen3",
		moss: "MOSS",
		piper: "Piper"
	}, r = X.ttsMode === "piper" ? "Legacy" : Ns(Ds.value?.quality || "experimental");
	return `${t.map((e) => n[e] || e).join(" + ") || "Unavailable"} · ${r}`;
}), ks = Y(() => !!(X.training.running || X.training.exit_code !== null || X.training.log_lines?.length)), As = Y(() => X.samples[X.sampleBucket] || []), js = Y(() => !!X.auto.trainer_link?.linked), Ms = Y(() => {
	let e = X.auto.stt_engines;
	return Array.isArray(e) && e.length ? e : [{
		id: "faster_whisper",
		label: "Faster Whisper"
	}, {
		id: "parakeet_onnx",
		label: "Parakeet ONNX"
	}];
});
function Ns(e) {
	return String(e || "").replaceAll("_", " ").replace(/\b\w/g, (e) => e.toUpperCase());
}
function Z(e) {
	return e ? X.busy.has(e) : X.busy.size > 0;
}
function Q(e, t) {
	t ? X.busy.add(e) : X.busy.delete(e);
}
function $(e, t = "success") {
	X.toast = {
		message: String(e || ""),
		tone: t,
		serial: X.toast.serial + 1
	};
}
function Ps(e, t) {
	$(e instanceof Error ? e.message : t, "error");
}
function Fs(e) {
	X.session = e || {}, Array.isArray(e.available_languages) && e.available_languages.length && (X.languages = e.available_languages), Array.isArray(e.available_english_accents) && e.available_english_accents.length && (X.englishAccents = e.available_english_accents), e.raw_phrase && (X.phrase = e.raw_phrase), e.language && (X.language = e.language), e.english_accent && (X.englishAccent = e.english_accent), e.tts_mode && (X.ttsMode = e.tts_mode), e.training && (X.training = e.training);
}
async function Is() {
	let e = await hs("/api/session");
	return Fs(e), e;
}
async function Ls() {
	if (!X.phrase.trim()) {
		$("Enter a wake phrase first.", "warning");
		return;
	}
	Q("session", !0);
	try {
		let e = await gs("/api/start_session", {
			phrase: X.phrase.trim(),
			language: X.language,
			english_accent: X.englishAccent,
			tts_mode: X.ttsMode
		});
		Fs(e), $(`Session ${e.safe_word || "started"} is ready.`);
	} catch (e) {
		Ps(e, "Session failed to start.");
	} finally {
		Q("session", !1);
	}
}
async function Rs() {
	let e = !!X.training.running;
	if (!(e && !window.confirm("Training is running. Stop training cleanly and end this session?"))) {
		Q("session", !0), ws &&= (window.clearInterval(ws), 0);
		try {
			Fs(await gs("/api/stop_session")), $(e ? "Training stopped cleanly and the session ended." : "Session ended. You can edit the wake phrase now.");
		} catch (t) {
			e && sc(), Ps(t, "Session could not be stopped.");
		} finally {
			Q("session", !1);
		}
	}
}
function zs() {
	if (!X.phrase.trim() || !("speechSynthesis" in window)) return;
	let e = new SpeechSynthesisUtterance(X.phrase.trim());
	e.lang = X.language, window.speechSynthesis.cancel(), window.speechSynthesis.speak(e);
}
function Bs() {
	let e = Ds.value?.engines || [], t = e.some((e) => e !== "piper"), n = e.includes("piper");
	X.ttsMode === "modern" && !t && (X.ttsMode = "piper"), X.ttsMode === "hybrid" && !(t && n) && (X.ttsMode = t ? "modern" : "piper"), X.ttsMode === "piper" && !n && (X.ttsMode = "modern"), (X.language !== "en" || X.ttsMode === "piper") && (X.englishAccent = "mixed");
}
async function Vs(e = !1) {
	e || Q("samples", !0);
	try {
		let e = await hs("/api/samples");
		X.samples = {
			...ys(),
			...e
		};
		for (let e of ["personal", "negative"]) {
			let t = Math.max(0, Math.ceil((X.samples[e]?.length || 0) / 50) - 1);
			X.samplePage[e] = Math.min(X.samplePage[e], t);
		}
		return e;
	} finally {
		e || Q("samples", !1);
	}
}
async function Hs(e = !1) {
	e || Q("captured", !0);
	try {
		let e = await hs("/api/captured_audio");
		return X.captured = {
			...bs(),
			...e
		}, e;
	} finally {
		e || Q("captured", !1);
	}
}
function Us(e) {
	let t = e.target;
	X.selectedFiles = Array.from(t.files || []);
}
function Ws(e, t, n) {
	return new Promise((r, i) => {
		let a = new XMLHttpRequest(), o = new FormData();
		o.append("file", e, e.name), a.open("POST", "/api/upload_personal_sample"), a.responseType = "json", a.upload.onprogress = (r) => {
			r.lengthComputable && (X.uploadProgress = Math.round((t + r.loaded / r.total) / n * 100), X.uploadLabel = `Uploading ${e.name} (${t + 1}/${n})`, X.uploadDetail = "Sending and normalizing the recording.");
		}, a.onload = () => {
			let t = a.response || {};
			a.status >= 200 && a.status < 300 ? r(t) : i(Error(t.error || `Upload failed for ${e.name}`));
		}, a.onerror = () => i(/* @__PURE__ */ Error(`Upload failed for ${e.name}`)), a.send(o);
	});
}
async function Gs(e) {
	if (!X.session.safe_word) {
		$("Start a trainer session before uploading samples.", "warning");
		return;
	}
	if (X.selectedFiles.length) {
		Q("upload", !0);
		try {
			let t = [...X.selectedFiles];
			for (let e = 0; e < t.length; e += 1) await Ws(t[e], e, t.length);
			X.uploadProgress = 100, X.uploadLabel = "Upload complete", X.uploadDetail = `${t.length} sample${t.length === 1 ? "" : "s"} saved in the required training format.`, X.selectedFiles = [], e && (e.value = ""), await Promise.all([Is(), Vs(!0)]), $("Personal samples uploaded.");
		} catch (e) {
			X.uploadProgress = 0, Ps(e, "Sample upload failed.");
		} finally {
			Q("upload", !1);
		}
	}
}
async function Ks(e, t) {
	if (!(t === "discard" && !window.confirm(`Discard ${e.saved_as} from the captured-audio inbox?`))) {
		Q("review", !0);
		try {
			await gs(`/api/captured_audio/${encodeURIComponent(e.saved_as)}/${t}`), await Promise.all([
				Is(),
				Hs(!0),
				Vs(!0)
			]), $(t === "approve_personal" ? "Clip added to personal samples." : t === "mark_negative" ? "Clip marked negative." : "Clip discarded.");
		} catch (e) {
			Ps(e, "Review action failed.");
		} finally {
			Q("review", !1);
		}
	}
}
async function qs(e, t) {
	if (window.confirm(`Remove ${e.saved_as} from ${t} samples?`)) {
		Q("review", !0);
		try {
			await ms(`/api/samples/${t}/${encodeURIComponent(e.saved_as)}`, { method: "DELETE" }), await Vs(!0), $("Sample removed.");
		} catch (e) {
			Ps(e, "Sample removal failed.");
		} finally {
			Q("review", !1);
		}
	}
}
async function Js(e, t) {
	if (!window.confirm(`Revert ${e.saved_as} to its pre-trim version?`)) return;
	let n = new FormData();
	n.append("bucket", t), n.append("file_name", e.saved_as), Q("review", !0);
	try {
		await ms("/api/samples/revert", {
			method: "POST",
			body: n
		}), await Vs(!0), $("Original sample restored.");
	} catch (e) {
		Ps(e, "Sample revert failed.");
	} finally {
		Q("review", !1);
	}
}
async function Ys(e) {
	let t = e === "personal" ? Ts.value : Es.value;
	if (!(!t || !window.confirm(`Clear ${t} ${e} sample${t === 1 ? "" : "s"}?`))) {
		Q("review", !0);
		try {
			await gs(e === "personal" ? "/api/reset_recordings" : "/api/reset_negative_samples"), await Promise.all([
				Is(),
				Vs(!0),
				Hs(!0)
			]), $(`${Ns(e)} samples cleared.`);
		} catch (e) {
			Ps(e, "Samples could not be cleared.");
		} finally {
			Q("review", !1);
		}
	}
}
function Xs(e, t) {
	X.auto = e || {}, t && (X.autoForm = {
		...Ss(),
		...e.config || {}
	}, Array.isArray(e.config?.generated_negative_phrases) && (X.autoForm.generated_negative_phrases = e.config.generated_negative_phrases.join("\n")), X.autoForm.wake_phrase || (X.autoForm.wake_phrase = X.session.raw_phrase || ""), X.autoForm.language || (X.autoForm.language = X.session.language || "en"), X.autoForm.english_accent || (X.autoForm.english_accent = X.session.english_accent || "mixed"), String(X.autoForm.language).toLowerCase().startsWith("en") || (X.autoForm.english_accent = "mixed"));
}
async function Zs(e = !1) {
	let t = await hs("/api/auto_train");
	return Xs(t, e), t;
}
async function Qs() {
	if (![
		X.autoForm.train_microwakeword,
		X.autoForm.train_openwakeword,
		X.autoForm.train_nanowakeword_cnn,
		X.autoForm.train_nanowakeword_gru
	].some(Boolean)) {
		$("Select at least one model for Auto Training.", "warning");
		return;
	}
	Q("auto", !0);
	try {
		let e = await _s("/api/auto_train", X.autoForm);
		Xs(e, !0), $(e.config?.enabled ? "Auto Training saved and enabled." : "Auto Training saved.");
	} catch (e) {
		Ps(e, "Auto Training settings failed to save.");
	} finally {
		Q("auto", !1);
	}
}
async function $s(e) {
	Q("auto", !0);
	try {
		let t = await gs("/api/auto_train/action", { action: e });
		Xs(t, !1), e === "train_now" && (X.consoleOpen = !0, sc()), $(e === "review_now" ? `${Number(t.queued || 0)} clips queued for review.` : e === "train_now" ? "Training started." : "Wake word published.");
	} catch (e) {
		Ps(e, "Auto Training action failed.");
	} finally {
		Q("auto", !1);
	}
}
async function ec(e, t) {
	Q("link", !0);
	try {
		return await gs("/api/tater_link/claim", {
			tater_url: e.trim(),
			pairing_code: t.trim()
		}), X.autoForm.tater_url = e.trim(), await Zs(!1), $("Trainer linked securely to Tater."), !0;
	} catch (e) {
		return Ps(e, "Tater link failed."), !1;
	} finally {
		Q("link", !1);
	}
}
async function tc() {
	if (window.confirm("Unlink this trainer from Tater?")) {
		Q("auto", !0);
		try {
			await gs("/api/tater_link/unlink"), await Zs(!1), $("Trainer unlinked from Tater.", "warning");
		} catch (e) {
			Ps(e, "Tater unlink failed.");
		} finally {
			Q("auto", !1);
		}
	}
}
async function nc(e = !1) {
	e || Q("firmware", !0);
	try {
		let e = await hs("/api/trained_wake_words/catalog");
		X.wakeWords = Array.isArray(e.wake_words) ? e.wake_words : [];
	} finally {
		e || Q("firmware", !1);
	}
}
async function rc() {
	Q("data", !0);
	try {
		let e = await hs("/api/data");
		return X.managedData = {
			...xs(),
			...e
		}, e;
	} finally {
		Q("data", !1);
	}
}
async function ic(e) {
	if (!e.file_count) return;
	let t = `${dc(e.size_bytes)} · ${Number(e.file_count).toLocaleString()} file${e.file_count === 1 ? "" : "s"}`, n = e.rebuild_note ? `\n\n${e.rebuild_note}` : "";
	if (window.confirm(`Permanently delete ${e.label} (${t})?${n}\n\nThis cannot be undone.`)) {
		Q("data-delete", !0);
		try {
			let t = await ms(`/api/data/${encodeURIComponent(e.id)}`, { method: "DELETE" });
			X.managedData = {
				...xs(),
				...t
			}, await Promise.allSettled([
				Is(),
				Vs(!0),
				Hs(!0),
				nc(!0)
			]), $(`${e.label} deleted. ${dc(e.size_bytes)} released.`);
		} catch (t) {
			Ps(t, `${e.label} could not be deleted.`);
		} finally {
			Q("data-delete", !1);
		}
	}
}
async function ac(e) {
	try {
		await navigator.clipboard.writeText(e), $("Wake-word JSON URL copied.");
	} catch (e) {
		Ps(e, "Clipboard unavailable.");
	}
}
async function oc() {
	if (![
		X.trainMicroWakeWord,
		X.trainOpenWakeWord,
		X.trainNanoWakeWordCnn,
		X.trainNanoWakeWordGru
	].some(Boolean)) {
		$("Select at least one wake-word model to train.", "warning");
		return;
	}
	await Promise.all([Is(), Vs(!0)]);
	let e = !1;
	if (!(!Ts.value && (e = window.confirm("No positive samples are saved. Train anyway without personal voices?"), !e))) {
		Q("training-start", !0), X.training = {
			running: !0,
			exit_code: null,
			log_lines: ["Waiting for training output…"]
		}, X.consoleOpen = !0;
		try {
			await gs("/api/train", {
				allow_no_personal: e,
				train_microwakeword: X.trainMicroWakeWord,
				train_openwakeword: X.trainOpenWakeWord,
				train_nanowakeword_cnn: X.trainNanoWakeWordCnn,
				train_nanowakeword_gru: X.trainNanoWakeWordGru,
				generated_negative_phrases: X.generatedNegativePhrases
			}), sc();
		} catch (e) {
			X.training = {
				running: !1,
				exit_code: 1,
				log_lines: [e instanceof Error ? e.message : String(e)]
			}, Ps(e, "Training could not start.");
		} finally {
			Q("training-start", !1);
		}
	}
}
function sc() {
	if (ws) return;
	let e = async () => {
		try {
			let e = await hs("/api/train_status");
			X.training = {
				...vs(),
				...e.training || {}
			}, X.training.running || (window.clearInterval(ws), ws = 0, await Promise.all([Vs(!0), nc(!0)]), $(X.training.exit_code === 0 ? "Training finished successfully." : `Training ended with exit ${X.training.exit_code}.`, X.training.exit_code === 0 ? "success" : "error"));
		} catch {}
	};
	e(), ws = window.setInterval(() => void e(), 1500);
}
async function cc() {
	Q("bootstrap", !0);
	try {
		await Promise.allSettled([
			Is(),
			Vs(!0),
			Hs(!0),
			Zs(!0),
			nc(!0)
		]), Bs();
		try {
			let e = await hs("/api/train_status");
			X.training = {
				...vs(),
				...e.training || {}
			}, X.training.running && (X.consoleOpen = !0, sc());
		} catch {}
		Cs = window.setInterval(() => {
			X.activeView === "auto" && !Z("auto") && Zs(!1).catch(() => void 0);
		}, 2500), X.initialized = !0;
	} finally {
		Q("bootstrap", !1);
	}
}
function lc() {
	window.clearInterval(Cs), window.clearInterval(ws), Cs = 0, ws = 0;
}
function uc(e) {
	if (!e) return "";
	let t = new Date(String(e));
	return Number.isNaN(t.getTime()) ? String(e) : t.toLocaleString();
}
function dc(e) {
	let t = Math.max(0, Number(e) || 0);
	if (t < 1024) return `${Math.round(t)} B`;
	let n = [
		"KB",
		"MB",
		"GB",
		"TB"
	], r = t / 1024, i = n[0];
	for (let e = 1; e < n.length && r >= 1024; e += 1) r /= 1024, i = n[e];
	return `${r >= 10 ? r.toFixed(1) : r.toFixed(2)} ${i}`;
}
function fc(e) {
	if (!e) return "16 kHz · mono · 16-bit WAV";
	let t = Number(e.sample_rate || e.sample_rate_hz || 16e3), n = Number(e.channels || 1) === 1 ? "mono" : `${e.channels} channels`, r = Number(e.bits_per_sample || e.sample_width_bits || 16);
	return `${Math.round(t / 1e3)} kHz · ${n} · ${r}-bit`;
}
function pc(e) {
	if (e.blocked_by_vad) return {
		label: "Blocked by VAD",
		tone: "warning"
	};
	let t = String(e.event_type || "").toLowerCase();
	return t.includes("close") ? {
		label: e.capture_label || "Close miss",
		tone: "warning"
	} : t.includes("false") ? {
		label: e.capture_label || "False trigger",
		tone: "error"
	} : t.includes("wake") || t.includes("detect") ? {
		label: e.capture_label || "Wake trigger",
		tone: "success"
	} : {
		label: e.capture_label || "Captured",
		tone: "neutral"
	};
}
function mc(e, t) {
	return e.audio_url || `/api/audio/${t}/${encodeURIComponent(e.saved_as)}`;
}
//#endregion
//#region src/components/AudioTrimModal.vue?vue&type=script&setup=true&lang.ts
var hc = {
	class: "modal trim-modal",
	role: "dialog",
	"aria-modal": "true",
	"aria-label": "Trim audio"
}, gc = { class: "modal-head" }, _c = {
	key: 0,
	class: "empty-state"
}, vc = { class: "range-grid" }, yc = ["max"], bc = ["min", "max"], xc = { class: "row space" }, Sc = { class: "pill" }, Cc = {
	key: 0,
	class: "pill success"
}, wc = { class: "row modal-actions" }, Tc = ["disabled"], Ec = /* @__PURE__ */ fr({
	__name: "AudioTrimModal",
	setup(e) {
		let t = /* @__PURE__ */ Gt(null), n = /* @__PURE__ */ Gt(null), r = /* @__PURE__ */ Gt(0), i = /* @__PURE__ */ Gt(0), a = /* @__PURE__ */ Gt(0), o = /* @__PURE__ */ Gt([]), s = /* @__PURE__ */ Gt(!1), c = /* @__PURE__ */ Gt(!1);
		Nn(() => X.trimItem, async (e) => {
			if (!e) {
				n.value = null;
				return;
			}
			s.value = !0;
			try {
				let t = `/api/audio/${encodeURIComponent(X.trimBucket)}/${encodeURIComponent(e.saved_as)}`, s = await fetch(t);
				if (!s.ok) throw Error("Audio could not be loaded.");
				let c = new (window.AudioContext || window.webkitAudioContext)();
				n.value = await c.decodeAudioData(await s.arrayBuffer()), r.value = n.value.duration, i.value = 0, a.value = r.value, await c.close();
				try {
					let t = await ms(`/api/samples/${encodeURIComponent(X.trimBucket)}/${encodeURIComponent(e.saved_as)}/vad`, { method: "POST" });
					o.value = Array.isArray(t.segments) ? t.segments : [], o.value.length && (i.value = Math.max(0, Number(o.value[0].start || 0)), a.value = Math.min(r.value, Number(o.value[0].end || r.value)));
				} catch {
					o.value = [];
				}
				await hn(), d();
			} catch (e) {
				$(e instanceof Error ? e.message : "Audio could not be loaded.", "error"), l();
			} finally {
				s.value = !1;
			}
		}, { immediate: !0 }), Nn([i, a], () => d());
		function l() {
			X.trimItem = null, n.value = null, o.value = [];
		}
		function u() {
			let e = o.value[0];
			e && (i.value = Number(e.start), a.value = Number(e.end));
		}
		function d() {
			let e = t.value, s = n.value;
			if (!e || !s || !r.value) return;
			let c = e.getBoundingClientRect();
			if (!c.width || !c.height) return;
			let l = window.devicePixelRatio || 1;
			e.width = Math.round(c.width * l), e.height = Math.round(c.height * l);
			let u = e.getContext("2d");
			if (!u) return;
			u.scale(l, l);
			let d = c.width, f = c.height, p = f / 2, m = s.getChannelData(0), h = Math.max(1, Math.floor(m.length / d));
			u.clearRect(0, 0, d, f), u.strokeStyle = "rgba(222, 218, 212, .24)", u.lineWidth = 1, u.beginPath();
			for (let e = 0; e < d; e += 1) {
				let t = 1, n = -1;
				for (let r = 0; r < h; r += 1) {
					let i = m[Math.floor(e) * h + r] || 0;
					t = Math.min(t, i), n = Math.max(n, i);
				}
				u.moveTo(e, p + t * p * .84), u.lineTo(e, p + n * p * .84);
			}
			u.stroke();
			let g = i.value / r.value * d, _ = a.value / r.value * d;
			u.fillStyle = "rgba(8, 8, 9, .66)", u.fillRect(0, 0, g, f), u.fillRect(_, 0, d - _, f), u.fillStyle = "rgba(255, 145, 52, .12)", u.fillRect(g, 0, _ - g, f), u.strokeStyle = "#ff9134", u.lineWidth = 2;
			for (let e of [g, _]) u.beginPath(), u.moveTo(e, 0), u.lineTo(e, f), u.stroke();
			u.strokeStyle = "rgba(68, 225, 165, .55)";
			for (let e of o.value) {
				let t = e.start / r.value * d;
				u.beginPath(), u.moveTo(t, 0), u.lineTo(t, f), u.stroke();
			}
		}
		function f() {
			let e = n.value;
			if (!e) return;
			let t = new (window.AudioContext || window.webkitAudioContext)(), r = t.createBufferSource();
			r.buffer = e, r.connect(t.destination), r.start(0, i.value, Math.max(.01, a.value - i.value)), r.onended = () => void t.close();
		}
		async function p() {
			let e = n.value;
			if (!e) throw Error("Audio is not loaded.");
			let t = Math.floor(i.value * e.sampleRate), r = Math.min(Math.floor(a.value * e.sampleRate), e.length), o = 16e3, s;
			if (e.sampleRate === o) s = e.getChannelData(0).slice(t, r);
			else {
				let n = Math.max(1, Math.floor((r - t) * o / e.sampleRate)), c = new OfflineAudioContext(1, n, o), l = c.createBufferSource();
				l.buffer = e, l.connect(c.destination), l.start(0, i.value, a.value - i.value), s = (await c.startRendering()).getChannelData(0);
			}
			let c = /* @__PURE__ */ new ArrayBuffer(44 + s.length * 2), l = new DataView(c);
			l.setUint32(0, 1380533830, !1), l.setUint32(4, 36 + s.length * 2, !0), l.setUint32(8, 1463899717, !1), l.setUint32(12, 1718449184, !1), l.setUint32(16, 16, !0), l.setUint16(20, 1, !0), l.setUint16(22, 1, !0), l.setUint32(24, o, !0), l.setUint32(28, o * 2, !0), l.setUint16(32, 2, !0), l.setUint16(34, 16, !0), l.setUint32(36, 1684108385, !1), l.setUint32(40, s.length * 2, !0);
			for (let e = 0; e < s.length; e += 1) l.setInt16(44 + e * 2, Math.max(-32768, Math.min(32767, Math.round(s[e] * 32767))), !0);
			return new Blob([c], { type: "audio/wav" });
		}
		async function m() {
			let e = X.trimItem;
			if (e) {
				c.value = !0;
				try {
					let t = new FormData();
					t.append("file", await p(), "trimmed.wav"), t.append("bucket", X.trimBucket), t.append("source_file", e.saved_as), t.append("start_time", i.value.toFixed(3)), t.append("end_time", a.value.toFixed(3));
					let n = await ms("/api/samples/trim", {
						method: "POST",
						body: t
					});
					l(), await Vs(!0), $(n.message || "Trimmed sample saved.");
				} catch (e) {
					$(e instanceof Error ? e.message : "Trim failed.", "error");
				} finally {
					c.value = !1;
				}
			}
		}
		function h() {
			X.trimItem && d();
		}
		return window.addEventListener("resize", h), Ar(() => window.removeEventListener("resize", h)), (e, n) => (W(), na(Jn, { to: "body" }, [I(X).trimItem ? (W(), G("div", {
			key: 0,
			class: "modal-backdrop",
			onClick: as(l, ["self"])
		}, [K("section", hc, [
			K("header", gc, [K("div", null, [n[2] ||= K("span", { class: "eyebrow" }, "Audio editor", -1), K("h2", null, "Trim " + A(I(X).trimItem.saved_as), 1)]), K("button", {
				type: "button",
				class: "button ghost",
				onClick: l
			}, "Close")]),
			n[3] ||= K("p", { class: "muted" }, "Keep the spoken wake phrase and remove excess silence or noise. VAD markers appear in green.", -1),
			s.value ? (W(), G("div", _c, "Loading waveform…")) : (W(), G(H, { key: 1 }, [
				K("canvas", {
					ref_key: "canvas",
					ref: t,
					class: "waveform"
				}, null, 512),
				K("div", vc, [K("label", null, [K("span", null, "Start · " + A(i.value.toFixed(2)) + "s", 1), R(K("input", {
					"onUpdate:modelValue": n[0] ||= (e) => i.value = e,
					type: "range",
					min: "0",
					max: Math.max(0, a.value - .01),
					step: ".01"
				}, null, 8, yc), [[
					Xo,
					i.value,
					void 0,
					{ number: !0 }
				]])]), K("label", null, [K("span", null, "End · " + A(a.value.toFixed(2)) + "s", 1), R(K("input", {
					"onUpdate:modelValue": n[1] ||= (e) => a.value = e,
					type: "range",
					min: Math.min(r.value, i.value + .01),
					max: r.value,
					step: ".01"
				}, null, 8, bc), [[
					Xo,
					a.value,
					void 0,
					{ number: !0 }
				]])])]),
				K("div", xc, [K("span", Sc, "Selection " + A(Math.max(0, a.value - i.value).toFixed(2)) + "s", 1), o.value.length ? (W(), G("span", Cc, A(o.value.length) + " speech segment" + A(o.value.length === 1 ? "" : "s"), 1)) : J("", !0)]),
				K("div", wc, [
					K("button", {
						type: "button",
						onClick: f
					}, "Play selection"),
					o.value.length ? (W(), G("button", {
						key: 0,
						type: "button",
						onClick: u
					}, "Select first VAD")) : J("", !0),
					K("button", {
						type: "button",
						class: "button primary",
						disabled: c.value,
						onClick: m
					}, A(c.value ? "Saving…" : "Save trim"), 9, Tc)
				])
			], 64))
		])])) : J("", !0)]));
	}
}), Dc = { class: "app-shell" }, Oc = { class: "app-header" }, kc = { class: "header-status" }, Ac = {
	key: 0,
	class: "session-chip"
}, jc = {
	class: "tabs",
	"aria-label": "Trainer areas"
}, Mc = ["onClick"], Nc = { class: "tab-full" }, Pc = { class: "tab-short" }, Fc = { key: 0 }, Ic = { class: "main-content" }, Lc = {
	key: 0,
	class: "loading-panel"
}, Rc = { class: "panel" }, zc = { class: "panel-head" }, Bc = { class: "form-grid phrase-form" }, Vc = { class: "field wide" }, Hc = ["disabled"], Uc = { class: "field" }, Wc = ["disabled"], Gc = ["value"], Kc = {
	key: 0,
	class: "field"
}, qc = ["disabled"], Jc = ["value"], Yc = { class: "field" }, Xc = ["disabled"], Zc = ["disabled"], Qc = ["disabled"], $c = ["disabled"], el = { class: "row form-actions" }, tl = ["disabled"], nl = ["disabled"], rl = ["disabled"], il = { class: "panel" }, al = { class: "panel-head" }, ol = { class: "stats" }, sl = { class: "model-picker" }, cl = {
	class: "model-choice-grid",
	role: "group",
	"aria-label": "Wake-word models to train"
}, ll = ["aria-pressed"], ul = ["aria-pressed"], dl = ["aria-pressed"], fl = {
	key: 0,
	class: "nano-architecture-panel"
}, pl = { class: "nano-architecture-heading" }, ml = {
	class: "nano-architecture-grid",
	role: "group",
	"aria-label": "NanoWakeWord architectures"
}, hl = ["aria-pressed"], gl = ["aria-pressed"], _l = {
	key: 1,
	class: "hard-negative-card"
}, vl = { class: "hard-negative-head" }, yl = { class: "hard-negative-engines" }, bl = { class: "field" }, xl = {
	key: 0,
	class: "training-summary-card"
}, Sl = { class: "training-summary-grid" }, Cl = {
	key: 0,
	class: "summary-metrics"
}, wl = { class: "train-action" }, Tl = ["disabled"], El = { class: "panel-footer" }, Dl = ["disabled"], Ol = { class: "hero auto-hero" }, kl = { class: "panel" }, Al = { class: "toggle-list" }, jl = { class: "form-grid" }, Ml = { class: "field" }, Nl = { class: "field" }, Pl = {
	key: 0,
	class: "field"
}, Fl = ["value"], Il = { class: "field wide" }, Ll = ["value"], Rl = { class: "field" }, zl = { class: "panel" }, Bl = { class: "form-grid" }, Vl = { class: "field" }, Hl = { class: "field" }, Ul = { class: "model-picker" }, Wl = {
	class: "model-choice-grid",
	role: "group",
	"aria-label": "Auto Training wake-word models"
}, Gl = ["aria-pressed"], Kl = ["aria-pressed"], ql = ["aria-pressed"], Jl = {
	key: 0,
	class: "nano-architecture-panel"
}, Yl = { class: "nano-architecture-heading" }, Xl = {
	class: "nano-architecture-grid",
	role: "group",
	"aria-label": "Auto Training NanoWakeWord architectures"
}, Zl = ["aria-pressed"], Ql = ["aria-pressed"], $l = {
	key: 1,
	class: "hard-negative-card"
}, eu = { class: "hard-negative-head" }, tu = { class: "hard-negative-engines" }, nu = { class: "field" }, ru = {
	key: 0,
	class: "training-summary-card"
}, iu = { class: "training-summary-grid" }, au = {
	key: 0,
	class: "summary-metrics"
}, ou = { class: "stats" }, su = { class: "format-value" }, cu = { class: "format-value" }, lu = { class: "panel" }, uu = { class: "form-grid" }, du = { class: "field wide" }, fu = { class: "field wide" }, pu = { class: "link-row" }, mu = ["disabled"], hu = ["disabled"], gu = { class: "toggle-list compact" }, _u = { class: "panel action-panel" }, vu = { class: "action-grid" }, yu = ["disabled"], bu = ["disabled"], xu = ["disabled"], Su = ["disabled"], Cu = { class: "audit" }, wu = { class: "hero capture-hero" }, Tu = { class: "panel" }, Eu = { class: "panel-head" }, Du = ["disabled"], Ou = { class: "stats" }, ku = { class: "panel" }, Au = {
	key: 0,
	class: "empty-state"
}, ju = {
	key: 1,
	class: "audio-list"
}, Mu = {
	key: 0,
	class: "meta-row"
}, Nu = {
	key: 1,
	class: "transcript"
}, Pu = {
	key: 2,
	class: "transcript"
}, Fu = ["src"], Iu = ["disabled", "onClick"], Lu = ["disabled", "onClick"], Ru = ["disabled", "onClick"], zu = { class: "hero samples-hero" }, Bu = { class: "pill hero-pill" }, Vu = { class: "panel" }, Hu = { class: "panel-head sample-head" }, Uu = { class: "segment-control" }, Wu = { class: "row toolbar" }, Gu = ["disabled"], Ku = ["disabled"], qu = ["disabled"], Ju = {
	key: 0,
	class: "empty-state"
}, Yu = {
	key: 1,
	class: "audio-list compact-list"
}, Xu = { class: "row" }, Zu = {
	key: 0,
	class: "pill warning"
}, Qu = {
	key: 0,
	class: "transcript"
}, $u = {
	key: 1,
	class: "transcript"
}, ed = ["src"], td = ["onClick"], nd = ["onClick"], rd = ["disabled", "onClick"], id = {
	key: 2,
	class: "pagination"
}, ad = ["disabled"], od = ["disabled"], sd = { class: "panel" }, cd = { class: "dropzone" }, ld = ["disabled"], ud = { class: "progress-card" }, dd = { class: "progress-track" }, fd = { class: "hero data-hero" }, pd = { class: "pill hero-pill" }, md = { class: "panel" }, hd = { class: "panel-head" }, gd = ["disabled"], _d = { class: "stats" }, vd = { class: "format-value" }, yd = {
	key: 0,
	class: "data-warning"
}, bd = { class: "panel-head" }, xd = { class: "number" }, Sd = { class: "data-list" }, Cd = { class: "data-copy" }, wd = { class: "data-title" }, Td = {
	key: 0,
	class: "data-note"
}, Ed = { class: "data-usage" }, Dd = ["disabled", "onClick"], Od = {
	key: 0,
	class: "panel empty-state"
}, kd = { class: "hero firmware-hero" }, Ad = { class: "panel" }, jd = { class: "panel-head" }, Md = ["disabled"], Nd = {
	key: 0,
	class: "empty-state"
}, Pd = {
	key: 1,
	class: "word-list"
}, Fd = ["href"], Id = {
	key: 1,
	class: "muted"
}, Ld = ["href"], Rd = ["href"], zd = ["href"], Bd = { class: "meta-row" }, Vd = { key: 0 }, Hd = { key: 1 }, Ud = { key: 2 }, Wd = ["disabled", "onClick"], Gd = { class: "panel compatibility-panel" }, Kd = {
	key: 0,
	class: "empty-state"
}, qd = {
	key: 1,
	class: "word-list"
}, Jd = ["href"], Yd = {
	key: 1,
	class: "muted"
}, Xd = ["disabled", "onClick"], Zd = {
	class: "modal console-modal",
	role: "dialog",
	"aria-modal": "true",
	"aria-label": "Training console"
}, Qd = { class: "modal-head" }, $d = { class: "row console-actions" }, ef = {
	class: "modal link-modal",
	role: "dialog",
	"aria-modal": "true",
	"aria-label": "Link Tater"
}, tf = { class: "modal-head" }, nf = {
	key: 0,
	class: "link-success"
}, rf = {
	key: 1,
	class: "stack"
}, af = { class: "field" }, of = { class: "field" }, sf = ["disabled"], cf = "/static/images/tater-wake-word-trainer.png", lf = 50, uf = /* @__PURE__ */ fr({
	__name: "TrainerApp",
	setup(e) {
		let t = /* @__PURE__ */ Gt(null), n = /* @__PURE__ */ Gt(null), r = /* @__PURE__ */ Gt(!0), i = /* @__PURE__ */ Gt(""), a = /* @__PURE__ */ Gt(""), o = /* @__PURE__ */ Gt(!1), s = [
			{
				id: "trainer",
				label: "Trainer",
				short: "Train"
			},
			{
				id: "auto",
				label: "Auto Training",
				short: "Auto"
			},
			{
				id: "firmware",
				label: "Wake Words",
				short: "Words"
			},
			{
				id: "captured",
				label: "Captured Audio",
				short: "Inbox"
			},
			{
				id: "samples",
				label: "Samples",
				short: "Samples"
			},
			{
				id: "data",
				label: "Data",
				short: "Data"
			}
		], c = Y(() => {
			let e = X.samplePage[X.sampleBucket];
			return As.value.slice(e * lf, (e + 1) * lf);
		}), l = Y(() => Math.max(1, Math.ceil(As.value.length / lf))), u = Y(() => X.auto.state || {}), d = Y(() => X.auto.runtime || {}), f = Y(() => {
			let e = u.value, t = [];
			return e.last_review_result && t.push(`Last review: ${String(e.last_review_result).replaceAll("_", " ")}`), e.last_review_file && t.push(String(e.last_review_file)), e.last_review_transcript && t.push(`STT: “${e.last_review_transcript}”`), e.last_review_error && t.push(`Error: ${e.last_review_error}`), e.last_stt_engine && t.push(`STT engine: ${String(e.last_stt_engine).replaceAll("_", " ")}`), e.last_notify_at && t.push(e.last_notify_error ? `Publish failed: ${e.last_notify_error}` : `Wake word published ${uc(e.last_notify_at)}`), t.join(" · ") || "No automatic review has run yet.";
		}), p = Y(() => X.training.running ? {
			text: "Training running",
			tone: "warning"
		} : X.training.exit_code === 0 ? {
			text: "Training finished",
			tone: "success"
		} : X.training.exit_code === null ? {
			text: "Not started",
			tone: "neutral"
		} : {
			text: `Exit ${X.training.exit_code}`,
			tone: "error"
		}), m = Y(() => d.value.review_running ? {
			text: `Transcribing ${d.value.review_file || "wake"}`,
			tone: "warning"
		} : X.training.running && X.auto.config?.enabled ? {
			text: "Training running",
			tone: "warning"
		} : X.auto.config?.enabled ? {
			text: "Enabled",
			tone: "success"
		} : {
			text: "Disabled",
			tone: "neutral"
		}), h = Y(() => X.training.log_lines?.length ? X.training.log_lines : ["No training output yet."]), g = Y(() => {
			let e = /* @__PURE__ */ new Map();
			for (let t of X.managedData.items || []) {
				let n = e.get(t.category) || [];
				n.push(t), e.set(t.category, n);
			}
			return Array.from(e, ([e, t]) => ({
				name: e,
				items: t
			}));
		}), _ = Y(() => X.trainNanoWakeWordCnn || X.trainNanoWakeWordGru), v = Y(() => X.autoForm.train_nanowakeword_cnn || X.autoForm.train_nanowakeword_gru), y = Y(() => [
			...X.trainMicroWakeWord ? ["MWW"] : [],
			...X.trainOpenWakeWord ? ["OWW"] : [],
			..._.value ? ["NWW"] : []
		]), b = Y(() => [
			...X.autoForm.train_microwakeword ? ["MWW"] : [],
			...X.autoForm.train_openwakeword ? ["OWW"] : [],
			...v.value ? ["NWW"] : []
		]);
		Nn([() => X.language, () => X.ttsMode], Bs), Nn(() => X.toast.serial, () => window.setTimeout(() => {
			X.toast.message = "";
		}, 4500)), Nn(h, async () => {
			r.value && (await hn(), r.value && n.value && (n.value.scrollTop = n.value.scrollHeight));
		}), Nn(() => X.consoleOpen, async (e) => {
			e && (r.value = !0, await hn(), C());
		}), Dr(() => {
			cc(), document.addEventListener("keydown", x);
		}), Ar(() => {
			lc(), document.removeEventListener("keydown", x);
		});
		function x(e) {
			e.key === "Escape" && (X.consoleOpen = !1, X.taterLinkOpen = !1, X.trimItem = null);
		}
		function S() {
			let e = n.value;
			if (!e) return;
			let t = e.scrollHeight - e.clientHeight - e.scrollTop;
			r.value = t <= 32;
		}
		function C() {
			let e = n.value;
			e && (r.value = !0, e.scrollTop = e.scrollHeight);
		}
		function w(e) {
			X.activeView = e, (e === "auto" ? Zs(!1) : e === "captured" ? Hs() : e === "samples" ? Vs() : e === "firmware" ? nc() : e === "data" ? rc() : Promise.resolve()).catch((e) => $(e instanceof Error ? e.message : "Refresh failed.", "error"));
		}
		function ee(e) {
			X.sampleBucket = e;
		}
		function te(e, t) {
			X.trimBucket = t, X.trimItem = e;
		}
		function ne() {
			_.value ? (X.trainNanoWakeWordCnn = !1, X.trainNanoWakeWordGru = !1) : X.trainNanoWakeWordCnn = !0;
		}
		function T() {
			v.value ? (X.autoForm.train_nanowakeword_cnn = !1, X.autoForm.train_nanowakeword_gru = !1) : X.autoForm.train_nanowakeword_cnn = !0;
		}
		function re(e) {
			return new Set(String(e || "").split(/[\n,]+/).map((e) => e.trim().toLocaleLowerCase()).filter(Boolean)).size;
		}
		function E() {
			i.value = X.autoForm.tater_url || "http://127.0.0.1:8501", a.value = "", o.value = !1, X.taterLinkOpen = !0, hn(() => document.querySelector("#pairing-code")?.focus());
		}
		function ie() {
			let e = a.value.toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 8);
			a.value = e.length > 4 ? `${e.slice(0, 4)}-${e.slice(4)}` : e;
		}
		async function ae() {
			if (!i.value.trim() || !a.value.trim()) {
				$("Tater address and pairing code are required.", "warning");
				return;
			}
			o.value = await ec(i.value, a.value);
		}
		function D(e) {
			let t = [];
			return e.source_device && t.push(String(e.source_device)), e.wake_word && t.push(String(e.wake_word)), e.max_probability !== null && e.max_probability !== void 0 && t.push(`max ${e.max_probability}`), e.average_probability !== null && e.average_probability !== void 0 && t.push(`avg ${e.average_probability}`), e.detection_profile && t.push(`profile ${String(e.detection_profile).replaceAll("_", " ")}`), e.auto_review_status && t.push(`auto ${String(e.auto_review_status).replaceAll("_", " ")}`), e.vad_max_probability !== null && e.vad_max_probability !== void 0 && t.push(`VAD ${e.vad_max_probability}`), t;
		}
		function oe(e) {
			let t = [];
			e.original_name && e.original_name !== e.saved_as && t.push(`From ${e.original_name}`);
			let n = uc(e.reviewed_at || e.received_at || e.created_at);
			return n && t.push(`Saved ${n}`), e.message && t.push(String(e.message)), e.auto_negative && t.push("Auto-reviewed false positive"), e.auto_positive && t.push("Auto-promoted close miss"), t.join(" · ") || "Training sample";
		}
		function O(e) {
			return String(e.json_url || e.url || e.jsonUrl || "");
		}
		function se(e) {
			return String(e.esphome_json_url || e.esphomeJsonUrl || "");
		}
		function ce(e) {
			return String(e.model_url || e.modelUrl || "");
		}
		function le(e) {
			return String(e.openwakeword_model_url || "");
		}
		function ue(e) {
			return e.nanowakeword_models && typeof e.nanowakeword_models == "object" ? e.nanowakeword_models : {};
		}
		function fe(e) {
			let t = e.trim().toLowerCase();
			return /^(✓|✅)|success|finished/.test(t) ? "success" : /^(✗|❌)|error|failed|traceback/.test(t) ? "error" : /^(⚠|warning)/.test(t) ? "warning" : /^={4,}|^-----|^=====/.test(t) ? "heading" : "";
		}
		return (e, d) => (W(), G("div", Dc, [
			d[160] ||= K("div", {
				class: "ambient ambient-one",
				"aria-hidden": "true"
			}, null, -1),
			d[161] ||= K("div", {
				class: "ambient ambient-two",
				"aria-hidden": "true"
			}, null, -1),
			K("header", Oc, [K("div", { class: "brand" }, [K("div", {
				class: "brand-mark",
				"aria-hidden": "true"
			}, [K("img", {
				src: cf,
				alt: ""
			})]), d[56] ||= K("div", null, [
				K("span", { class: "eyebrow" }, "Tater tools"),
				K("h1", null, "Wake Word Studio"),
				K("p", null, "Generate voices, curate real recordings, train, and publish.")
			], -1)]), K("div", kc, [d[57] ||= K("span", { class: "live-dot" }, [K("i"), ua("Local trainer")], -1), I(X).session.safe_word ? (W(), G("span", Ac, A(I(X).session.safe_word) + " · " + A(I(X).language), 1)) : J("", !0)])]),
			K("nav", jc, [(W(), G(H, null, z(s, (e) => K("button", {
				key: e.id,
				type: "button",
				class: k({ active: I(X).activeView === e.id }),
				onClick: (t) => w(e.id)
			}, [
				K("span", Nc, A(e.label), 1),
				K("span", Pc, A(e.short), 1),
				e.id === "captured" && I(X).captured.captured_count ? (W(), G("b", Fc, A(I(X).captured.captured_count), 1)) : J("", !0)
			], 10, Mc)), 64))]),
			K("main", Ic, [I(X).initialized ? (W(), G(H, { key: 1 }, [I(X).activeView === "trainer" ? (W(), G(H, { key: 0 }, [
				d[86] ||= da("<section class=\"hero training-hero\"><div><span class=\"eyebrow\">Training studio</span><h2>Build a personal wake word</h2><p>Choose a multilingual voice route, check your real samples, then follow the model pipeline live.</p></div><div class=\"step-row\"><span><b>1</b> Phrase</span><span><b>2</b> Samples</span><span><b>3</b> Train</span></div></section>", 1),
				K("section", Rc, [
					K("header", zc, [
						d[59] ||= K("div", { class: "number" }, "1", -1),
						d[60] ||= K("div", null, [K("h3", null, "Phrase + voice"), K("p", null, "The phrase and voice route lock while a session is active.")], -1),
						K("span", { class: k(["pill", I(X).session.safe_word ? "success" : ""]) }, A(I(X).session.safe_word ? `Session · ${I(X).session.safe_word}` : "No session"), 3)
					]),
					K("div", Bc, [
						K("label", Vc, [d[61] ||= K("span", null, "Wake phrase", -1), R(K("input", {
							"onUpdate:modelValue": d[0] ||= (e) => I(X).phrase = e,
							type: "text",
							placeholder: "e.g. \"hey tater\"",
							disabled: !!I(X).session.safe_word || I(Z)("session"),
							onKeyup: d[1] ||= ss((...e) => I(Ls) && I(Ls)(...e), ["enter"])
						}, null, 40, Hc), [[Xo, I(X).phrase]])]),
						K("label", Uc, [
							d[62] ||= K("span", null, "Language", -1),
							R(K("select", {
								"onUpdate:modelValue": d[2] ||= (e) => I(X).language = e,
								disabled: !!I(X).session.safe_word || I(Z)("session")
							}, [(W(!0), G(H, null, z(I(X).languages, (e) => (W(), G("option", {
								key: e.code,
								value: e.code
							}, A(e.label), 9, Gc))), 128))], 8, Wc), [[$o, I(X).language]]),
							K("small", null, A(I(Os)), 1)
						]),
						I(X).language === "en" && I(X).ttsMode !== "piper" ? (W(), G("label", Kc, [
							d[63] ||= K("span", null, "English accent emphasis", -1),
							R(K("select", {
								"onUpdate:modelValue": d[3] ||= (e) => I(X).englishAccent = e,
								disabled: !!I(X).session.safe_word || I(Z)("session")
							}, [(W(!0), G(H, null, z(I(X).englishAccents, (e) => (W(), G("option", {
								key: e.code,
								value: e.code
							}, A(e.label), 9, Jc))), 128))], 8, qc), [[$o, I(X).englishAccent]]),
							d[64] ||= K("small", null, "Qwen emphasizes this accent; MOSS carries it through accepted references. OmniVoice and Piper keep broad English coverage.", -1)
						])) : J("", !0),
						K("label", Yc, [
							d[65] ||= K("span", null, "TTS source", -1),
							R(K("select", {
								"onUpdate:modelValue": d[4] ||= (e) => I(X).ttsMode = e,
								disabled: !!I(X).session.safe_word || I(Z)("session")
							}, [
								K("option", {
									value: "hybrid",
									disabled: !I(X).languages.find((e) => e.code === I(X).language)?.engines?.includes("piper")
								}, "Four-provider ensemble · recommended", 8, Zc),
								K("option", {
									value: "modern",
									disabled: !I(X).languages.find((e) => e.code === I(X).language)?.engines?.some((e) => e !== "piper")
								}, "Modern only · no Piper", 8, Qc),
								K("option", {
									value: "piper",
									disabled: !I(X).languages.find((e) => e.code === I(X).language)?.engines?.includes("piper")
								}, "Piper only · legacy", 8, $c)
							], 8, Xc), [[$o, I(X).ttsMode]]),
							d[66] ||= K("small", null, "Models download once and stay cached.", -1)
						])
					]),
					K("div", el, [I(X).session.safe_word ? (W(), G("button", {
						key: 1,
						type: "button",
						class: "button danger",
						disabled: I(Z)("session"),
						onClick: d[6] ||= (...e) => I(Rs) && I(Rs)(...e)
					}, A(I(Z)("session") ? "Stopping…" : I(X).training.running ? "Stop session + training" : "Stop session"), 9, nl)) : (W(), G("button", {
						key: 0,
						type: "button",
						class: "button primary",
						disabled: I(Z)("session") || !I(X).phrase.trim(),
						onClick: d[5] ||= (...e) => I(Ls) && I(Ls)(...e)
					}, A(I(Z)("session") ? "Starting…" : "Start session"), 9, tl)), K("button", {
						type: "button",
						disabled: !I(X).phrase.trim(),
						onClick: d[7] ||= (...e) => I(zs) && I(zs)(...e)
					}, "System preview", 8, rl)])
				]),
				K("section", il, [
					K("header", al, [
						d[67] ||= K("div", { class: "number" }, "2", -1),
						d[68] ||= K("div", null, [K("h3", null, "Train wake word"), K("p", null, "Personal positives and reviewed false-wake negatives are automatically included.")], -1),
						K("span", { class: k(["pill", p.value.tone]) }, A(p.value.text), 3)
					]),
					K("div", ol, [
						K("article", null, [d[69] ||= K("span", null, "Positive samples", -1), K("strong", null, A(I(Ts)), 1)]),
						K("article", null, [d[70] ||= K("span", null, "Negative samples", -1), K("strong", null, A(I(Es)), 1)]),
						d[71] ||= K("article", null, [K("span", null, "Training format"), K("strong", { class: "format-value" }, "16 kHz · mono · WAV")], -1)
					]),
					K("div", sl, [
						K("div", cl, [
							K("button", {
								type: "button",
								class: k(["model-choice", { active: I(X).trainMicroWakeWord }]),
								"aria-pressed": I(X).trainMicroWakeWord,
								onClick: d[8] ||= (e) => I(X).trainMicroWakeWord = !I(X).trainMicroWakeWord
							}, [...d[72] ||= [
								K("i", null, "MWW", -1),
								K("span", null, [K("strong", null, "MicroWakeWord"), K("small", null, "TFLite for ESP32 satellites and ESPHome.")], -1),
								K("b", null, "✓", -1)
							]], 10, ll),
							K("button", {
								type: "button",
								class: k(["model-choice", { active: I(X).trainOpenWakeWord }]),
								"aria-pressed": I(X).trainOpenWakeWord,
								onClick: d[9] ||= (e) => I(X).trainOpenWakeWord = !I(X).trainOpenWakeWord
							}, [...d[73] ||= [
								K("i", null, "OWW", -1),
								K("span", null, [K("strong", null, "openWakeWord"), K("small", null, "ONNX for satellites with OWW support.")], -1),
								K("b", null, "✓", -1)
							]], 10, ul),
							K("button", {
								type: "button",
								class: k(["model-choice", { active: _.value }]),
								"aria-pressed": _.value,
								onClick: ne
							}, [...d[74] ||= [
								K("i", null, "NWW", -1),
								K("span", null, [K("strong", null, "NanoWakeWord"), K("small", null, "ONNX with selectable CNN and GRU builds.")], -1),
								K("b", null, "✓", -1)
							]], 10, dl)
						]),
						_.value ? (W(), G("div", fl, [K("div", pl, [d[75] ||= K("div", null, [K("strong", null, "NanoWakeWord architectures"), K("small", null, "Select either architecture or train both.")], -1), K("span", null, A(I(X).trainNanoWakeWordCnn && I(X).trainNanoWakeWordGru ? "2 selected" : "1 selected"), 1)]), K("div", ml, [K("button", {
							type: "button",
							class: k(["architecture-choice", { active: I(X).trainNanoWakeWordCnn }]),
							"aria-pressed": I(X).trainNanoWakeWordCnn,
							onClick: d[10] ||= (e) => I(X).trainNanoWakeWordCnn = !I(X).trainNanoWakeWordCnn
						}, [...d[76] ||= [
							K("i", null, "CNN", -1),
							K("span", null, [K("strong", null, "CNN"), K("small", null, "Recommended default with stronger recall.")], -1),
							K("b", null, "✓", -1)
						]], 10, hl), K("button", {
							type: "button",
							class: k(["architecture-choice", { active: I(X).trainNanoWakeWordGru }]),
							"aria-pressed": I(X).trainNanoWakeWordGru,
							onClick: d[11] ||= (e) => I(X).trainNanoWakeWordGru = !I(X).trainNanoWakeWordGru
						}, [...d[77] ||= [
							K("i", null, "GRU", -1),
							K("span", null, [K("strong", null, "GRU"), K("small", null, "Recurrent model tuned for fewer false triggers.")], -1),
							K("b", null, "✓", -1)
						]], 10, gl)])])) : J("", !0),
						y.value.length ? (W(), G("section", _l, [
							K("header", vl, [
								d[78] ||= K("div", { class: "hard-negative-icon" }, "−", -1),
								d[79] ||= K("div", null, [
									K("span", { class: "eyebrow" }, "Optional precision tuning"),
									K("strong", null, "Hard negative training"),
									K("small", null, "Teach the selected models which similar-sounding phrases must not wake the assistant.")
								], -1),
								K("b", null, A(re(I(X).generatedNegativePhrases) ? `${re(I(X).generatedNegativePhrases)} phrase${re(I(X).generatedNegativePhrases) === 1 ? "" : "s"}` : "Optional"), 1)
							]),
							K("div", yl, [d[80] ||= K("span", null, "Applied to", -1), (W(), G(H, null, z([
								"MWW",
								"OWW",
								"NWW"
							], (e) => K("i", {
								key: e,
								class: k({ active: y.value.includes(e) })
							}, A(e), 3)), 64))]),
							K("label", bl, [
								d[81] ||= K("span", null, "Phrases to reject", -1),
								R(K("textarea", {
									"onUpdate:modelValue": d[12] ||= (e) => I(X).generatedNegativePhrases = e,
									rows: "4",
									placeholder: "hey later\nhey waiter\nokay tater"
								}, null, 512), [[Xo, I(X).generatedNegativePhrases]]),
								d[82] ||= K("small", null, "One phrase per line. Each selected engine synthesizes and validates these near-misses alongside reviewed recordings and its broad negative data.", -1)
							]),
							d[83] ||= K("footer", null, [K("span", null, "Unchanged phrases reuse their cached audio on later runs."), K("strong", null, "The actual wake phrase is ignored automatically.")], -1)
						])) : J("", !0)
					]),
					!I(X).training.running && I(X).training.summary?.models?.length ? (W(), G("section", xl, [K("header", null, [
						K("div", { class: k(["summary-mark", I(X).training.summary.tone]) }, A(I(X).training.summary.tone === "success" ? "✓" : I(X).training.summary.tone === "error" ? "×" : "!"), 3),
						K("div", null, [
							d[84] ||= K("span", { class: "eyebrow" }, "Last completed run", -1),
							K("h4", null, A(I(X).training.summary.wake_word), 1),
							K("p", null, [ua(A(I(X).training.summary.message), 1), I(X).training.summary.completed_at ? (W(), G(H, { key: 0 }, [ua(" · " + A(I(uc)(I(X).training.summary.completed_at)), 1)], 64)) : J("", !0)])
						]),
						K("span", { class: k(["pill", I(X).training.summary.tone]) }, A(I(X).training.summary.overall), 3)
					]), K("div", Sl, [(W(!0), G(H, null, z(I(X).training.summary.models, (e) => (W(), G("article", {
						key: e.id,
						class: k(e.tone)
					}, [
						K("header", null, [K("div", null, [K("span", null, A(e.id.replaceAll("_", " ")), 1), K("strong", null, A(e.label), 1)]), K("b", null, A(e.rating), 1)]),
						K("p", null, A(e.message), 1),
						e.metrics?.length ? (W(), G("div", Cl, [(W(!0), G(H, null, z(e.metrics, (e) => (W(), G("span", { key: `${e.label}-${e.value}` }, [K("small", null, A(e.label), 1), K("strong", null, A(e.value), 1)]))), 128))])) : J("", !0)
					], 2))), 128))])])) : J("", !0),
					K("div", wl, [K("button", {
						type: "button",
						class: "button primary large",
						disabled: !I(X).session.safe_word || I(X).training.running || I(Z)("training-start"),
						onClick: d[13] ||= (...e) => I(oc) && I(oc)(...e)
					}, A(I(X).training.running ? "Training in progress" : "Start training"), 9, Tl)]),
					K("footer", El, [d[85] ||= K("span", null, "Training opens the console automatically and continues if the window is closed.", -1), K("button", {
						type: "button",
						disabled: !I(ks),
						onClick: d[14] ||= (e) => I(X).consoleOpen = !0
					}, "Open console", 8, Dl)])
				])
			], 64)) : I(X).activeView === "auto" ? (W(), G(H, { key: 1 }, [
				K("section", Ol, [d[87] ||= K("div", null, [
					K("span", { class: "eyebrow" }, "False-positive loop"),
					K("h2", null, "Auto Training"),
					K("p", null, "Transcribe captures, sort negatives, recover close misses, retrain on schedule, and publish through Tater.")
				], -1), K("span", { class: k(["pill hero-pill", m.value.tone]) }, A(m.value.text), 3)]),
				K("section", kl, [
					d[97] ||= K("header", { class: "panel-head" }, [K("div", { class: "number" }, "1"), K("div", null, [K("h3", null, "Review rules"), K("p", null, "Conservative local STT keeps uncertain clips in the manual inbox.")])], -1),
					K("div", Al, [
						K("label", null, [R(K("input", {
							"onUpdate:modelValue": d[15] ||= (e) => I(X).autoForm.enabled = e,
							type: "checkbox"
						}, null, 512), [[Zo, I(X).autoForm.enabled]]), d[88] ||= K("span", null, [K("strong", null, "Enable Auto Training"), K("small", null, "Queue eligible wake triggers for local transcription.")], -1)]),
						K("label", null, [R(K("input", {
							"onUpdate:modelValue": d[16] ||= (e) => I(X).autoForm.delete_confirmed_wakes = e,
							type: "checkbox"
						}, null, 512), [[Zo, I(X).autoForm.delete_confirmed_wakes]]), d[89] ||= K("span", null, [K("strong", null, "Delete confirmed good wakes"), K("small", null, "Remove normal triggers when STT confirms the phrase.")], -1)]),
						K("label", null, [R(K("input", {
							"onUpdate:modelValue": d[17] ||= (e) => I(X).autoForm.promote_close_misses = e,
							type: "checkbox"
						}, null, 512), [[Zo, I(X).autoForm.promote_close_misses]]), d[90] ||= K("span", null, [K("strong", null, "Promote confirmed close misses"), K("small", null, "Move verified close misses into positive samples.")], -1)])
					]),
					K("div", jl, [
						K("label", Ml, [d[91] ||= K("span", null, "Wake phrase", -1), R(K("input", {
							"onUpdate:modelValue": d[18] ||= (e) => I(X).autoForm.wake_phrase = e,
							type: "text"
						}, null, 512), [[Xo, I(X).autoForm.wake_phrase]])]),
						K("label", Nl, [d[92] ||= K("span", null, "STT language", -1), R(K("input", {
							"onUpdate:modelValue": d[19] ||= (e) => I(X).autoForm.language = e,
							type: "text"
						}, null, 512), [[Xo, I(X).autoForm.language]])]),
						String(I(X).autoForm.language).toLowerCase().startsWith("en") ? (W(), G("label", Pl, [
							d[93] ||= K("span", null, "English accent emphasis", -1),
							R(K("select", { "onUpdate:modelValue": d[20] ||= (e) => I(X).autoForm.english_accent = e }, [(W(!0), G(H, null, z(I(X).englishAccents, (e) => (W(), G("option", {
								key: e.code,
								value: e.code
							}, A(e.label), 9, Fl))), 128))], 512), [[$o, I(X).autoForm.english_accent]]),
							d[94] ||= K("small", null, "Used when Auto Training needs to regenerate English TTS.", -1)
						])) : J("", !0),
						K("label", Il, [
							d[95] ||= K("span", null, "STT engine", -1),
							R(K("select", { "onUpdate:modelValue": d[21] ||= (e) => I(X).autoForm.stt_engine = e }, [(W(!0), G(H, null, z(I(Ms), (e) => (W(), G("option", {
								key: e.id || e.value,
								value: e.id || e.value
							}, A(e.label || e.name || e.id), 9, Ll))), 128))], 512), [[$o, I(X).autoForm.stt_engine]]),
							K("small", null, A(I(Ms).find((e) => (e.id || e.value) === I(X).autoForm.stt_engine)?.description || "Runs locally on this trainer."), 1)
						]),
						K("label", Rl, [d[96] ||= K("span", null, "Minimum transcript characters", -1), R(K("input", {
							"onUpdate:modelValue": d[22] ||= (e) => I(X).autoForm.minimum_transcript_chars = e,
							min: "1",
							max: "100",
							type: "number"
						}, null, 512), [[
							Xo,
							I(X).autoForm.minimum_transcript_chars,
							void 0,
							{ number: !0 }
						]])])
					])
				]),
				K("section", zl, [
					d[117] ||= K("header", { class: "panel-head" }, [K("div", { class: "number" }, "2"), K("div", null, [K("h3", null, "Training schedule"), K("p", null, "A run starts only after enough newly reviewed negatives accumulate.")])], -1),
					K("div", Bl, [K("label", Vl, [d[99] ||= K("span", null, "Run training", -1), R(K("select", { "onUpdate:modelValue": d[23] ||= (e) => I(X).autoForm.schedule_hours = e }, [...d[98] ||= [
						K("option", { value: 0 }, "Manually only", -1),
						K("option", { value: 6 }, "Every 6 hours", -1),
						K("option", { value: 12 }, "Every 12 hours", -1),
						K("option", { value: 24 }, "Every day", -1),
						K("option", { value: 48 }, "Every 2 days", -1),
						K("option", { value: 168 }, "Every week", -1)
					]], 512), [[
						$o,
						I(X).autoForm.schedule_hours,
						void 0,
						{ number: !0 }
					]])]), K("label", Hl, [d[100] ||= K("span", null, "Minimum new negatives", -1), R(K("input", {
						"onUpdate:modelValue": d[24] ||= (e) => I(X).autoForm.minimum_new_negatives = e,
						min: "1",
						max: "10000",
						type: "number"
					}, null, 512), [[
						Xo,
						I(X).autoForm.minimum_new_negatives,
						void 0,
						{ number: !0 }
					]])])]),
					K("div", Ul, [
						K("div", Wl, [
							K("button", {
								type: "button",
								class: k(["model-choice", { active: I(X).autoForm.train_microwakeword }]),
								"aria-pressed": I(X).autoForm.train_microwakeword,
								onClick: d[25] ||= (e) => I(X).autoForm.train_microwakeword = !I(X).autoForm.train_microwakeword
							}, [...d[101] ||= [
								K("i", null, "MWW", -1),
								K("span", null, [K("strong", null, "MicroWakeWord"), K("small", null, "Refresh the ESP32 and ESPHome model.")], -1),
								K("b", null, "✓", -1)
							]], 10, Gl),
							K("button", {
								type: "button",
								class: k(["model-choice", { active: I(X).autoForm.train_openwakeword }]),
								"aria-pressed": I(X).autoForm.train_openwakeword,
								onClick: d[26] ||= (e) => I(X).autoForm.train_openwakeword = !I(X).autoForm.train_openwakeword
							}, [...d[102] ||= [
								K("i", null, "OWW", -1),
								K("span", null, [K("strong", null, "openWakeWord"), K("small", null, "Refresh the openWakeWord ONNX model.")], -1),
								K("b", null, "✓", -1)
							]], 10, Kl),
							K("button", {
								type: "button",
								class: k(["model-choice", { active: v.value }]),
								"aria-pressed": v.value,
								onClick: T
							}, [...d[103] ||= [
								K("i", null, "NWW", -1),
								K("span", null, [K("strong", null, "NanoWakeWord"), K("small", null, "Refresh selected CNN and GRU models.")], -1),
								K("b", null, "✓", -1)
							]], 10, ql)
						]),
						v.value ? (W(), G("div", Jl, [K("div", Yl, [d[104] ||= K("div", null, [K("strong", null, "NanoWakeWord architectures"), K("small", null, "Select either architecture or retrain both.")], -1), K("span", null, A(I(X).autoForm.train_nanowakeword_cnn && I(X).autoForm.train_nanowakeword_gru ? "2 selected" : "1 selected"), 1)]), K("div", Xl, [K("button", {
							type: "button",
							class: k(["architecture-choice", { active: I(X).autoForm.train_nanowakeword_cnn }]),
							"aria-pressed": I(X).autoForm.train_nanowakeword_cnn,
							onClick: d[27] ||= (e) => I(X).autoForm.train_nanowakeword_cnn = !I(X).autoForm.train_nanowakeword_cnn
						}, [...d[105] ||= [
							K("i", null, "CNN", -1),
							K("span", null, [K("strong", null, "CNN"), K("small", null, "Recommended default with stronger recall.")], -1),
							K("b", null, "✓", -1)
						]], 10, Zl), K("button", {
							type: "button",
							class: k(["architecture-choice", { active: I(X).autoForm.train_nanowakeword_gru }]),
							"aria-pressed": I(X).autoForm.train_nanowakeword_gru,
							onClick: d[28] ||= (e) => I(X).autoForm.train_nanowakeword_gru = !I(X).autoForm.train_nanowakeword_gru
						}, [...d[106] ||= [
							K("i", null, "GRU", -1),
							K("span", null, [K("strong", null, "GRU"), K("small", null, "Recurrent model tuned for fewer false triggers.")], -1),
							K("b", null, "✓", -1)
						]], 10, Ql)])])) : J("", !0),
						b.value.length ? (W(), G("section", $l, [
							K("header", eu, [
								d[107] ||= K("div", { class: "hard-negative-icon" }, "−", -1),
								d[108] ||= K("div", null, [
									K("span", { class: "eyebrow" }, "Optional precision tuning"),
									K("strong", null, "Hard negative training"),
									K("small", null, "Keep near-miss protection consistent whenever Auto Training refreshes a model.")
								], -1),
								K("b", null, A(re(I(X).autoForm.generated_negative_phrases) ? `${re(I(X).autoForm.generated_negative_phrases)} phrase${re(I(X).autoForm.generated_negative_phrases) === 1 ? "" : "s"}` : "Optional"), 1)
							]),
							K("div", tu, [d[109] ||= K("span", null, "Applied to", -1), (W(), G(H, null, z([
								"MWW",
								"OWW",
								"NWW"
							], (e) => K("i", {
								key: e,
								class: k({ active: b.value.includes(e) })
							}, A(e), 3)), 64))]),
							K("label", nu, [
								d[110] ||= K("span", null, "Phrases to reject", -1),
								R(K("textarea", {
									"onUpdate:modelValue": d[29] ||= (e) => I(X).autoForm.generated_negative_phrases = e,
									rows: "4",
									placeholder: "hey later\nhey waiter\nokay tater"
								}, null, 512), [[Xo, I(X).autoForm.generated_negative_phrases]]),
								d[111] ||= K("small", null, "One phrase per line. Saved with the schedule and synthesized for every selected model.", -1)
							]),
							d[112] ||= K("footer", null, [K("span", null, "Reviewed false wakes still receive the strongest model weight."), K("strong", null, "The actual wake phrase is ignored automatically.")], -1)
						])) : J("", !0)
					]),
					!I(X).training.running && I(X).training.summary?.models?.length ? (W(), G("section", ru, [K("header", null, [
						K("div", { class: k(["summary-mark", I(X).training.summary.tone]) }, A(I(X).training.summary.tone === "success" ? "✓" : I(X).training.summary.tone === "error" ? "×" : "!"), 3),
						K("div", null, [
							d[113] ||= K("span", { class: "eyebrow" }, "Last completed run", -1),
							K("h4", null, A(I(X).training.summary.wake_word), 1),
							K("p", null, [ua(A(I(X).training.summary.message), 1), I(X).training.summary.completed_at ? (W(), G(H, { key: 0 }, [ua(" · " + A(I(uc)(I(X).training.summary.completed_at)), 1)], 64)) : J("", !0)])
						]),
						K("span", { class: k(["pill", I(X).training.summary.tone]) }, A(I(X).training.summary.overall), 3)
					]), K("div", iu, [(W(!0), G(H, null, z(I(X).training.summary.models, (e) => (W(), G("article", {
						key: `auto-${e.id}`,
						class: k(e.tone)
					}, [
						K("header", null, [K("div", null, [K("span", null, A(e.id.replaceAll("_", " ")), 1), K("strong", null, A(e.label), 1)]), K("b", null, A(e.rating), 1)]),
						K("p", null, A(e.message), 1),
						e.metrics?.length ? (W(), G("div", au, [(W(!0), G(H, null, z(e.metrics, (e) => (W(), G("span", { key: `${e.label}-${e.value}` }, [K("small", null, A(e.label), 1), K("strong", null, A(e.value), 1)]))), 128))])) : J("", !0)
					], 2))), 128))])])) : J("", !0),
					K("div", ou, [
						K("article", null, [d[114] ||= K("span", null, "Pending negatives", -1), K("strong", null, A(Number(u.value.pending_negative_count || 0)), 1)]),
						K("article", null, [d[115] ||= K("span", null, "Next check", -1), K("strong", su, A(u.value.next_run_at ? I(uc)(u.value.next_run_at) : "Manual"), 1)]),
						K("article", null, [d[116] ||= K("span", null, "Last training", -1), K("strong", cu, A(u.value.last_train_finished_at ? I(uc)(u.value.last_train_finished_at) : "Never"), 1)])
					])
				]),
				K("section", lu, [
					d[121] ||= K("header", { class: "panel-head" }, [K("div", { class: "number" }, "3"), K("div", null, [K("h3", null, "Publish to Tater"), K("p", null, "Securely activate successful models across every connected satellite.")])], -1),
					K("div", uu, [K("label", du, [
						d[118] ||= K("span", null, "Trainer public URL", -1),
						R(K("input", {
							"onUpdate:modelValue": d[30] ||= (e) => I(X).autoForm.advertised_base_url = e,
							type: "text",
							placeholder: "Auto-detect LAN address"
						}, null, 512), [[Xo, I(X).autoForm.advertised_base_url]]),
						K("small", null, A(I(X).autoForm.advertised_base_url ? `Configured: ${I(X).autoForm.advertised_base_url}` : `Detected: ${I(X).auto.advertised_base_url || "unavailable"}`), 1)
					]), K("label", fu, [d[119] ||= K("span", null, "Tater URL", -1), R(K("input", {
						"onUpdate:modelValue": d[31] ||= (e) => I(X).autoForm.tater_url = e,
						type: "text"
					}, null, 512), [[Xo, I(X).autoForm.tater_url]])])]),
					K("div", pu, [
						K("span", { class: k(["pill", I(js) ? "success" : "warning"]) }, A(I(js) ? `Linked${I(X).auto.trainer_link?.tater_name ? ` · ${I(X).auto.trainer_link.tater_name}` : ""}` : "Not linked"), 3),
						K("button", {
							type: "button",
							class: "button primary",
							disabled: I(Z)("auto"),
							onClick: E
						}, A(I(js) ? "Relink Tater" : "Link Tater"), 9, mu),
						I(js) ? (W(), G("button", {
							key: 0,
							type: "button",
							class: "button danger",
							disabled: I(Z)("auto"),
							onClick: d[32] ||= (...e) => I(tc) && I(tc)(...e)
						}, "Unlink", 8, hu)) : J("", !0)
					]),
					K("div", gu, [K("label", null, [R(K("input", {
						"onUpdate:modelValue": d[33] ||= (e) => I(X).autoForm.notify_satellites = e,
						type: "checkbox"
					}, null, 512), [[Zo, I(X).autoForm.notify_satellites]]), d[120] ||= K("span", null, [K("strong", null, "Activate after successful training"), K("small", null, "Tater applies the new word globally.")], -1)])])
				]),
				K("section", _u, [K("div", vu, [
					K("button", {
						type: "button",
						class: "button primary",
						disabled: I(Z)("auto"),
						onClick: d[34] ||= (...e) => I(Qs) && I(Qs)(...e)
					}, "Save Auto Training", 8, yu),
					K("button", {
						type: "button",
						disabled: I(Z)("auto"),
						onClick: d[35] ||= (e) => I($s)("review_now")
					}, "Review inbox now", 8, bu),
					K("button", {
						type: "button",
						disabled: I(Z)("auto") || I(X).training.running,
						onClick: d[36] ||= (e) => I($s)("train_now")
					}, "Train now", 8, xu),
					K("button", {
						type: "button",
						disabled: I(Z)("auto") || !I(js),
						onClick: d[37] ||= (e) => I($s)("notify_now")
					}, "Publish current word", 8, Su)
				]), K("p", Cu, A(f.value), 1)])
			], 64)) : I(X).activeView === "captured" ? (W(), G(H, { key: 2 }, [
				K("section", wu, [d[122] ||= K("div", null, [
					K("span", { class: "eyebrow" }, "Capture review"),
					K("h2", null, "Captured Audio"),
					K("p", null, "Listen to clips from your satellites and turn every real-world event into a better model.")
				], -1), K("span", { class: k(["pill hero-pill", I(X).captured.captured_count ? "warning" : ""]) }, A(I(X).captured.captured_count ? `${I(X).captured.captured_count} waiting` : "Inbox idle"), 3)]),
				K("section", Tu, [K("header", Eu, [
					d[123] ||= K("div", { class: "number" }, "1", -1),
					d[124] ||= K("div", null, [K("h3", null, "Review queue"), K("p", null, "Approve good phrases, keep false positives as negatives, or discard noise.")], -1),
					K("button", {
						type: "button",
						disabled: I(Z)("captured"),
						onClick: d[38] ||= (e) => I(Hs)()
					}, A(I(Z)("captured") ? "Refreshing…" : "Refresh inbox"), 9, Du)
				]), K("div", Ou, [
					K("article", null, [d[125] ||= K("span", null, "Inbox", -1), K("strong", null, A(I(X).captured.captured_count), 1)]),
					K("article", null, [d[126] ||= K("span", null, "Reviewed negatives", -1), K("strong", null, A(I(Es)), 1)]),
					K("article", null, [d[127] ||= K("span", null, "Personal samples", -1), K("strong", null, A(I(Ts)), 1)])
				])]),
				K("section", ku, [d[130] ||= K("header", { class: "panel-head" }, [K("div", { class: "number" }, "2"), K("div", null, [K("h3", null, "Listen + sort"), K("p", null, "Metadata remains visible so borderline detections are easy to understand.")])], -1), I(X).captured.items?.length ? (W(), G("div", ju, [(W(!0), G(H, null, z(I(X).captured.items, (e) => (W(), G("article", {
					key: e.saved_as,
					class: "audio-card"
				}, [
					K("header", null, [K("div", null, [K("strong", null, A(e.original_name || e.saved_as), 1), K("small", null, A(I(uc)(e.captured_at || e.received_at)) + " " + A(e.message || ""), 1)]), K("span", { class: k(["pill", I(pc)(e).tone]) }, A(I(pc)(e).label), 3)]),
					D(e).length ? (W(), G("div", Mu, [(W(!0), G(H, null, z(D(e), (e) => (W(), G("span", { key: e }, A(e), 1))), 128))])) : J("", !0),
					e.transcript ? (W(), G("div", Nu, [d[128] ||= K("b", null, "STT", -1), ua(" " + A(e.transcript), 1)])) : J("", !0),
					e.auto_review_guided_transcript ? (W(), G("div", Pu, [d[129] ||= K("b", null, "Guided wake check", -1), ua(" " + A(e.auto_review_guided_transcript), 1)])) : J("", !0),
					K("audio", {
						controls: "",
						preload: "none",
						src: I(mc)(e, "captured")
					}, null, 8, Fu),
					K("footer", null, [K("span", null, A(e.saved_as) + " · " + A(I(fc)(e.final_format)), 1), K("div", null, [
						K("button", {
							type: "button",
							disabled: I(Z)("review"),
							onClick: (t) => I(Ks)(e, "approve_personal")
						}, "Add positive", 8, Iu),
						K("button", {
							type: "button",
							disabled: I(Z)("review"),
							onClick: (t) => I(Ks)(e, "mark_negative")
						}, "Mark negative", 8, Lu),
						K("button", {
							type: "button",
							class: "button danger ghost",
							disabled: I(Z)("review"),
							onClick: (t) => I(Ks)(e, "discard")
						}, "Discard", 8, Ru)
					])])
				]))), 128))])) : (W(), G("div", Au, "No captured audio yet. Clips sent by satellites will appear here."))])
			], 64)) : I(X).activeView === "samples" ? (W(), G(H, { key: 3 }, [
				K("section", zu, [d[131] ||= K("div", null, [
					K("span", { class: "eyebrow" }, "Sample library"),
					K("h2", null, "Current Training Samples"),
					K("p", null, "Audit positives and negatives, trim recordings precisely, and import seed audio.")
				], -1), K("span", Bu, A(I(Ts) + I(Es)) + " total", 1)]),
				K("section", Vu, [
					K("header", Hu, [
						d[134] ||= K("div", { class: "number" }, "1", -1),
						d[135] ||= K("div", null, [K("h3", null, "Saved samples"), K("p", null, "Personal clips are positives. Negative clips are false wakes and hard negatives.")], -1),
						K("div", Uu, [K("button", {
							type: "button",
							class: k({ active: I(X).sampleBucket === "personal" }),
							onClick: d[39] ||= (e) => ee("personal")
						}, [d[132] ||= ua("Personal ", -1), K("b", null, A(I(Ts)), 1)], 2), K("button", {
							type: "button",
							class: k({ active: I(X).sampleBucket === "negative" }),
							onClick: d[40] ||= (e) => ee("negative")
						}, [d[133] ||= ua("Negative ", -1), K("b", null, A(I(Es)), 1)], 2)])
					]),
					K("div", Wu, [
						K("button", {
							type: "button",
							disabled: I(Z)("samples"),
							onClick: d[41] ||= (e) => I(Vs)()
						}, "Refresh", 8, Gu),
						K("button", {
							type: "button",
							class: "button danger ghost",
							disabled: I(Z)("review") || I(Ts) === 0,
							onClick: d[42] ||= (e) => I(Ys)("personal")
						}, "Clear positives", 8, Ku),
						K("button", {
							type: "button",
							class: "button danger ghost",
							disabled: I(Z)("review") || I(Es) === 0,
							onClick: d[43] ||= (e) => I(Ys)("negative")
						}, "Clear negatives", 8, qu)
					]),
					I(As).length ? (W(), G("div", Yu, [(W(!0), G(H, null, z(c.value, (e) => (W(), G("article", {
						key: e.saved_as,
						class: "audio-card"
					}, [
						K("header", null, [K("div", null, [K("strong", null, A(e.saved_as), 1), K("small", null, A(oe(e)), 1)]), K("div", Xu, [e.trimmed ? (W(), G("span", Zu, "Trimmed")) : J("", !0), K("span", { class: k(["pill", I(X).sampleBucket === "personal" ? "success" : "error"]) }, A(I(X).sampleBucket === "personal" ? "Positive" : "Negative"), 3)])]),
						e.transcript ? (W(), G("div", Qu, [d[136] ||= K("b", null, "STT", -1), ua(" " + A(e.transcript), 1)])) : J("", !0),
						e.auto_review_guided_transcript ? (W(), G("div", $u, [d[137] ||= K("b", null, "Guided wake check", -1), ua(" " + A(e.auto_review_guided_transcript), 1)])) : J("", !0),
						K("audio", {
							controls: "",
							preload: "none",
							src: I(mc)(e, I(X).sampleBucket)
						}, null, 8, ed),
						K("footer", null, [K("span", null, A(I(fc)(e.final_format)), 1), K("div", null, [
							K("button", {
								type: "button",
								onClick: (t) => te(e, I(X).sampleBucket)
							}, "Trim", 8, td),
							e.trimmed ? (W(), G("button", {
								key: 0,
								type: "button",
								onClick: (t) => I(Js)(e, I(X).sampleBucket)
							}, "Revert", 8, nd)) : J("", !0),
							K("button", {
								type: "button",
								class: "button danger ghost",
								disabled: I(Z)("review"),
								onClick: (t) => I(qs)(e, I(X).sampleBucket)
							}, "Remove", 8, rd)
						])])
					]))), 128))])) : (W(), G("div", Ju, "No " + A(I(X).sampleBucket) + " samples saved yet.", 1)),
					l.value > 1 ? (W(), G("div", id, [
						K("button", {
							type: "button",
							disabled: I(X).samplePage[I(X).sampleBucket] === 0,
							onClick: d[44] ||= (e) => I(X).samplePage[I(X).sampleBucket]--
						}, "Previous", 8, ad),
						K("span", null, "Page " + A(I(X).samplePage[I(X).sampleBucket] + 1) + " of " + A(l.value), 1),
						K("button", {
							type: "button",
							disabled: I(X).samplePage[I(X).sampleBucket] >= l.value - 1,
							onClick: d[45] ||= (e) => I(X).samplePage[I(X).sampleBucket]++
						}, "Next", 8, od)
					])) : J("", !0)
				]),
				K("section", sd, [
					d[139] ||= K("header", { class: "panel-head" }, [K("div", { class: "number" }, "2"), K("div", null, [K("h3", null, "Manual sample import"), K("p", null, "Optional seed recordings are normalized to the trainer’s required WAV format.")])], -1),
					K("label", cd, [
						K("input", {
							ref_key: "uploadInput",
							ref: t,
							type: "file",
							multiple: "",
							accept: "audio/*,.wav,.mp3,.m4a,.flac,.ogg,.aac,.webm,.opus",
							onChange: d[46] ||= (...e) => I(Us) && I(Us)(...e)
						}, null, 544),
						d[138] ||= K("span", null, [K("strong", null, "Choose one or many audio files"), K("small", null, "WAV, MP3, M4A, FLAC, OGG, AAC, OPUS, and WEBM")], -1),
						K("b", null, A(I(X).selectedFiles.length ? `${I(X).selectedFiles.length} selected` : "Browse"), 1)
					]),
					K("button", {
						type: "button",
						class: "button primary",
						disabled: !I(X).session.safe_word || !I(X).selectedFiles.length || I(Z)("upload"),
						onClick: d[47] ||= (e) => I(Gs)(t.value)
					}, A(I(Z)("upload") ? "Uploading…" : "Upload selected samples"), 9, ld),
					K("div", ud, [
						K("div", null, [K("strong", null, A(I(X).uploadLabel), 1), K("span", null, A(I(X).uploadProgress) + "%", 1)]),
						K("div", dd, [K("i", { style: de({ width: `${I(X).uploadProgress}%` }) }, null, 4)]),
						K("small", null, A(I(X).uploadDetail), 1)
					])
				])
			], 64)) : I(X).activeView === "data" ? (W(), G(H, { key: 4 }, [
				K("section", fd, [d[140] ||= K("div", null, [
					K("span", { class: "eyebrow" }, "Local storage"),
					K("h2", null, "Data Management"),
					K("p", null, "See exactly what the trainer has downloaded, generated, recorded, and produced.")
				], -1), K("span", pd, A(I(dc)(I(X).managedData.total_size_bytes)) + " total", 1)]),
				K("section", md, [
					K("header", hd, [
						d[141] ||= K("div", { class: "number" }, "i", -1),
						d[142] ||= K("div", null, [K("h3", null, "Trainer storage"), K("p", null, "Deleting an item is permanent. Required downloads and generated caches will be rebuilt the next time training needs them.")], -1),
						K("button", {
							type: "button",
							disabled: I(Z)("data") || I(Z)("data-delete"),
							onClick: d[48] ||= (e) => I(rc)()
						}, A(I(Z)("data") ? "Scanning…" : "Refresh sizes"), 9, gd)
					]),
					K("div", _d, [
						K("article", null, [d[143] ||= K("span", null, "Space used", -1), K("strong", vd, A(I(dc)(I(X).managedData.total_size_bytes)), 1)]),
						K("article", null, [d[144] ||= K("span", null, "Files", -1), K("strong", null, A(Number(I(X).managedData.total_file_count || 0).toLocaleString()), 1)]),
						K("article", null, [d[145] ||= K("span", null, "Individual items", -1), K("strong", null, A(I(X).managedData.items.length), 1)])
					]),
					I(X).training.running ? (W(), G("p", yd, "Stop the active training session before deleting data.")) : J("", !0)
				]),
				(W(!0), G(H, null, z(g.value, (e, t) => (W(), G("section", {
					key: e.name,
					class: "panel data-panel"
				}, [K("header", bd, [K("div", xd, A(t + 1), 1), K("div", null, [K("h3", null, A(e.name), 1), K("p", null, A(e.items.length) + " separately managed item" + A(e.items.length === 1 ? "" : "s"), 1)])]), K("div", Sd, [(W(!0), G(H, null, z(e.items, (e) => (W(), G("article", {
					key: e.id,
					class: k(["data-row", { empty: !e.file_count }])
				}, [
					K("div", Cd, [
						K("div", wd, [K("strong", null, A(e.label), 1), K("code", null, A(e.location), 1)]),
						K("small", null, A(e.description), 1),
						e.rebuild_note ? (W(), G("span", Td, A(e.rebuild_note), 1)) : J("", !0)
					]),
					K("div", Ed, [K("strong", null, A(I(dc)(e.size_bytes)), 1), K("span", null, A(Number(e.file_count || 0).toLocaleString()) + " file" + A(e.file_count === 1 ? "" : "s"), 1)]),
					K("button", {
						type: "button",
						class: "button danger ghost",
						disabled: !e.file_count || I(X).training.running || I(Z)("data") || I(Z)("data-delete"),
						onClick: (t) => I(ic)(e)
					}, A(I(Z)("data-delete") ? "Please wait…" : "Delete"), 9, Dd)
				], 2))), 128))])]))), 128)),
				!I(Z)("data") && !I(X).managedData.items.length ? (W(), G("section", Od, "No managed trainer data was found.")) : J("", !0)
			], 64)) : I(X).activeView === "firmware" ? (W(), G(H, { key: 5 }, [
				K("section", kd, [d[146] ||= K("div", null, [
					K("span", { class: "eyebrow" }, "Wake-word catalog"),
					K("h2", null, "Trained Wake Words"),
					K("p", null, "Copy a local JSON package URL into Tater to switch every native satellite live.")
				], -1), K("span", { class: k(["pill hero-pill", I(X).wakeWords.length ? "success" : "warning"]) }, A(I(X).wakeWords.length ? `${I(X).wakeWords.length} trained` : "Catalog empty"), 3)]),
				d[151] ||= K("div", { class: "native-notice" }, [K("strong", null, "Tater Native"), K("span", null, "These packages include model metadata and a direct model URL for live satellite updates.")], -1),
				K("section", Ad, [K("header", jd, [
					d[147] ||= K("div", { class: "number" }, "v1", -1),
					d[148] ||= K("div", null, [K("h3", null, "Published model URLs"), K("p", null, "URLs stay local and are refreshed after each successful run.")], -1),
					K("button", {
						type: "button",
						disabled: I(Z)("firmware"),
						onClick: d[49] ||= (e) => I(nc)()
					}, "Refresh", 8, Md)
				]), I(X).wakeWords.length ? (W(), G("div", Pd, [(W(!0), G(H, null, z(I(X).wakeWords, (e) => (W(), G("article", { key: e.key || O(e) }, [K("div", null, [
					K("strong", null, A(e.label || e.name || "Trained wake word"), 1),
					O(e) ? (W(), G("a", {
						key: 0,
						href: O(e),
						target: "_blank",
						rel: "noreferrer"
					}, "Tater wake-word bundle · " + A(O(e)), 9, Fd)) : (W(), G("span", Id, "Bundle URL unavailable")),
					ce(e) ? (W(), G("a", {
						key: 2,
						href: ce(e),
						target: "_blank",
						rel: "noreferrer"
					}, "MicroWakeWord model · " + A(ce(e)), 9, Ld)) : J("", !0),
					le(e) ? (W(), G("a", {
						key: 3,
						href: le(e),
						target: "_blank",
						rel: "noreferrer"
					}, "openWakeWord model · " + A(le(e)), 9, Rd)) : J("", !0),
					(W(!0), G(H, null, z(ue(e), (e, t) => (W(), G("a", {
						key: t,
						href: e,
						target: "_blank",
						rel: "noreferrer"
					}, "NanoWakeWord " + A(String(t).toUpperCase()) + " model · " + A(e), 9, zd))), 128)),
					K("div", Bd, [
						(W(!0), G(H, null, z(e.available_engines || [], (e) => (W(), G("span", { key: e }, A(String(e).replaceAll("_", " ")), 1))), 128)),
						e.language ? (W(), G("span", Vd, A(e.language), 1)) : J("", !0),
						e.trained_at ? (W(), G("span", Hd, A(I(uc)(e.trained_at)), 1)) : J("", !0),
						e.recall === void 0 ? J("", !0) : (W(), G("span", Ud, "recall " + A(e.recall), 1))
					])
				]), K("button", {
					type: "button",
					disabled: !O(e),
					onClick: (t) => I(ac)(O(e))
				}, "Copy bundle URL", 8, Wd)]))), 128))])) : (W(), G("div", Nd, "Train a wake word and its package will appear here."))]),
				d[152] ||= K("div", { class: "native-notice esphome-notice" }, [K("strong", null, "ESPHome"), K("span", null, "Strict micro_wake_word manifest without Tater Native or calibration extensions.")], -1),
				K("section", Gd, [d[150] ||= K("header", { class: "panel-head" }, [K("div", { class: "number" }, "ESP"), K("div", null, [K("h3", null, "ESPHome JSON"), K("p", null, "Use this URL as the model in an ESPHome micro_wake_word configuration.")])], -1), I(X).wakeWords.length ? (W(), G("div", qd, [(W(!0), G(H, null, z(I(X).wakeWords, (e) => (W(), G("article", { key: `esphome-${e.key || se(e)}` }, [K("div", null, [
					K("strong", null, A(e.label || e.name || "Trained wake word"), 1),
					se(e) ? (W(), G("a", {
						key: 0,
						href: se(e),
						target: "_blank",
						rel: "noreferrer"
					}, "ESPHome JSON · " + A(se(e)), 9, Jd)) : (W(), G("span", Yd, "ESPHome package URL unavailable")),
					d[149] ||= K("div", { class: "meta-row" }, [K("span", null, "Schema v2"), K("span", null, "Same TFLite model")], -1)
				]), K("button", {
					type: "button",
					disabled: !se(e),
					onClick: (t) => I(ac)(se(e))
				}, "Copy ESPHome URL", 8, Xd)]))), 128))])) : (W(), G("div", Kd, "ESPHome links appear after a wake word is trained."))])
			], 64)) : J("", !0)], 64)) : (W(), G("div", Lc, [...d[58] ||= [K("span", { class: "spinner" }, null, -1), K("strong", null, "Connecting to the local trainer…", -1)]]))]),
			(W(), na(Jn, { to: "body" }, [I(X).consoleOpen ? (W(), G("div", {
				key: 0,
				class: "modal-backdrop console-backdrop",
				onClick: d[51] ||= as((e) => I(X).consoleOpen = !1, ["self"])
			}, [K("section", Zd, [K("header", Qd, [d[153] ||= K("div", null, [
				K("span", { class: "eyebrow" }, "Live pipeline"),
				K("h2", null, "Training Console"),
				K("p", null, "Closing this window does not interrupt training.")
			], -1), K("div", $d, [
				r.value ? J("", !0) : (W(), G("button", {
					key: 0,
					type: "button",
					class: "console-follow",
					onClick: C
				}, "Jump to latest")),
				K("span", { class: k(["pill", p.value.tone]) }, A(p.value.text), 3),
				K("button", {
					type: "button",
					onClick: d[50] ||= (e) => I(X).consoleOpen = !1
				}, "Close")
			])]), K("pre", {
				ref_key: "consoleLog",
				ref: n,
				class: "console-log",
				onScrollPassive: S
			}, [(W(!0), G(H, null, z(h.value, (e, t) => (W(), G("span", {
				key: `${t}-${e}`,
				class: k(fe(e))
			}, A(e), 3))), 128))], 544)])])) : J("", !0)])),
			(W(), na(Jn, { to: "body" }, [I(X).taterLinkOpen ? (W(), G("div", {
				key: 0,
				class: "modal-backdrop",
				onClick: d[55] ||= as((e) => I(X).taterLinkOpen = !1, ["self"])
			}, [K("section", ef, [K("header", tf, [K("div", null, [
				d[154] ||= K("span", { class: "eyebrow" }, "Secure pairing", -1),
				K("h2", null, A(o.value ? "Tater linked" : "Link Tater"), 1),
				K("p", null, A(o.value ? "This trainer can securely publish wake-word updates." : "Enter the short-lived code shown in Tater Voice Settings."), 1)
			]), K("button", {
				type: "button",
				onClick: d[52] ||= (e) => I(X).taterLinkOpen = !1
			}, "Close")]), o.value ? (W(), G("div", nf, [
				d[155] ||= K("i", null, "✓", -1),
				K("strong", null, "Successfully linked" + A(I(X).auto.trainer_link?.tater_name ? ` to ${I(X).auto.trainer_link.tater_name}` : ""), 1),
				d[156] ||= K("span", null, "The private link key is stored locally and is never displayed.", -1)
			])) : (W(), G("div", rf, [
				K("label", af, [d[157] ||= K("span", null, "Tater address", -1), R(K("input", {
					"onUpdate:modelValue": d[53] ||= (e) => i.value = e,
					type: "text"
				}, null, 512), [[Xo, i.value]])]),
				K("label", of, [d[158] ||= K("span", null, "Tater pairing code", -1), R(K("input", {
					id: "pairing-code",
					"onUpdate:modelValue": d[54] ||= (e) => a.value = e,
					class: "pairing-code",
					maxlength: "9",
					placeholder: "ABCD-EFGH",
					autocomplete: "off",
					onInput: ie
				}, null, 544), [[Xo, a.value]])]),
				d[159] ||= K("small", null, "In Tater, open Voice Settings → Wake Word Trainer → Link Trainer.", -1),
				K("button", {
					type: "button",
					class: "button primary",
					disabled: I(Z)("link"),
					onClick: ae
				}, A(I(Z)("link") ? "Linking securely…" : "Link Tater"), 9, sf)
			]))])])) : J("", !0)])),
			q(Ec),
			q($a, { name: "toast" }, {
				default: Dn(() => [I(X).toast.message ? (W(), G("div", {
					key: 0,
					class: k(["toast", I(X).toast.tone]),
					role: "status"
				}, A(I(X).toast.message), 3)) : J("", !0)]),
				_: 1
			})
		]));
	}
}), df = document.getElementById("trainer-app");
if (!df) throw Error("Missing #trainer-app mount point");
ds(uf).mount(df);
//#endregion
